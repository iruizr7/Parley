//! Session-only availability results shared by startup, editors and job setup.
//! No expiry timer: explicit changes or refreshes invalidate the target.
use super::config::ProcessingTarget;
use anyhow::Result;
use serde_json::Value;
use std::{
    collections::HashMap,
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, LazyLock, Mutex,
    },
};

#[derive(Default)]
struct Entry {
    invalid: AtomicBool,
    value: tokio::sync::Mutex<Option<Value>>,
}

#[derive(Default)]
struct Cache(Mutex<HashMap<(String, String), Arc<Entry>>>);
static CACHE: LazyLock<Cache> = LazyLock::new(Cache::default);

impl Cache {
    async fn get<F, Fut>(&self, target: &str, key: String, check: F) -> Result<Value>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Value>>,
    {
        loop {
            let entry = self
                .0
                .lock()
                .unwrap()
                .entry((target.to_owned(), key.clone()))
                .or_default()
                .clone();
            let mut value = entry.value.lock().await;
            if entry.invalid.load(Ordering::SeqCst) {
                continue;
            }
            if let Some(value) = value.as_ref() {
                return Ok(value.clone());
            }
            let result = check().await;
            // A download/update/refresh may finish while this check is running.
            // Its result must not become the post-change availability result.
            if entry.invalid.load(Ordering::SeqCst) {
                continue;
            }
            let result = result?;
            *value = Some(result.clone());
            return Ok(result);
        }
    }

    fn invalidate(&self, target: &str) {
        self.0.lock().unwrap().retain(|(id, _), entry| {
            if id != target {
                return true;
            }
            entry.invalid.store(true, Ordering::SeqCst);
            false
        });
    }
}

pub(crate) async fn get<F, Fut>(target: &ProcessingTarget, scope: Value, check: F) -> Result<Value>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<Value>>,
{
    let mut connection = serde_json::to_value(target)?;
    // Renaming a machine does not change its runtime or model availability.
    connection.as_object_mut().unwrap().remove("name");
    let key = serde_json::to_string(&(connection, scope))?;
    CACHE.get(&target.id, key, check).await
}

pub(crate) fn invalidate(target: &str) {
    CACHE.invalidate(target);
}

/// Invalidate on both sides of a mutation, including failed/partial changes.
pub(crate) struct ChangingTarget(String);
impl ChangingTarget {
    pub(crate) fn new(target: &str) -> Self {
        invalidate(target);
        Self(target.to_owned())
    }
}
impl Drop for ChangingTarget {
    fn drop(&mut self) {
        invalidate(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[tokio::test]
    async fn simultaneous_editors_share_check_and_changed_choices_check_again() {
        let cache = Cache::default();
        let calls = AtomicUsize::new(0);
        let check = || async {
            calls.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            Ok(Value::Bool(true))
        };
        let (a, b) = tokio::join!(
            cache.get("a", "model1".into(), check),
            cache.get("a", "model1".into(), check)
        );
        assert_eq!(a.unwrap(), b.unwrap());
        cache.get("a", "model1".into(), check).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        cache.get("a", "model2".into(), check).await.unwrap();
        cache.get("b", "model1".into(), check).await.unwrap();
        cache.invalidate("a");
        cache.get("b", "model1".into(), check).await.unwrap();
        cache.get("a", "model1".into(), check).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn invalidated_inflight_result_is_discarded() {
        let cache = Cache::default();
        let calls = AtomicUsize::new(0);
        let result = cache
            .get("a", "pipeline".into(), || async {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                if call == 0 {
                    cache.invalidate("a");
                }
                Ok(Value::from(call))
            })
            .await
            .unwrap();
        assert_eq!(result, Value::from(1));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn transport_errors_can_be_retried() {
        let cache = Cache::default();
        assert!(cache
            .get("a", "pipeline".into(), || async {
                anyhow::bail!("offline")
            })
            .await
            .is_err());
        assert_eq!(
            cache
                .get("a", "pipeline".into(), || async { Ok(Value::Bool(true)) })
                .await
                .unwrap(),
            Value::Bool(true)
        );
    }
}
