output "service_url" {
  value       = google_cloud_run_v2_service.api.uri
  description = "Public URL of the Cloud Run service"
}

output "bucket_name" {
  value       = google_storage_bucket.data.name
  description = "GCS bucket holding the model pickle and prediction archive"
}

output "deployer_service_account" {
  value       = google_service_account.deployer.email
  description = "Service account GitHub Actions impersonates"
}

output "workload_identity_provider" {
  value       = google_iam_workload_identity_pool_provider.github.name
  description = "Full WIF provider resource name (paste into GitHub Actions secrets)"
}

output "artifact_registry_uri" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo}"
  description = "Artifact Registry path for the container image"
}
