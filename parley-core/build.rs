fn main() {
    // sqlx::migrate! embeds SQL; rebuild whenever a migration changes.
    println!("cargo:rerun-if-changed=migrations");
}
