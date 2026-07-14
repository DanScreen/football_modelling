variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  default     = "europe-west2"
  description = "GCP region for Cloud Run, Artifact Registry, GCS"
}

variable "service_name" {
  type    = string
  default = "football-api"
}

variable "artifact_repo" {
  type    = string
  default = "football-api"
}

variable "bucket_suffix" {
  type        = string
  default     = "football-data"
  description = "Bucket name suffix; final name is <project_id>-<suffix>"
}

variable "github_repo" {
  type        = string
  description = "GitHub repository in the form 'owner/repo' (used for Workload Identity binding)"
}

variable "schedule_cron" {
  type        = string
  default     = "0 8 * * *"
  description = "Cron expression for daily archive job (default 08:00 daily)"
}

variable "schedule_tz" {
  type    = string
  default = "Europe/London"
}
