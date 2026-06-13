terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ----------------------------------------------------------------------
# Required APIs
# ----------------------------------------------------------------------
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudscheduler.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "iamcredentials.googleapis.com",
    "iam.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# ----------------------------------------------------------------------
# GCS bucket — model pickle + daily prediction archive
# ----------------------------------------------------------------------
resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-${var.bucket_suffix}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  lifecycle_rule {
    condition { age = 365 }
    action { type = "Delete" }
  }

  depends_on = [google_project_service.apis]
}

# ----------------------------------------------------------------------
# Artifact Registry for the container image
# ----------------------------------------------------------------------
resource "google_artifact_registry_repository" "containers" {
  location      = var.region
  repository_id = var.artifact_repo
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ----------------------------------------------------------------------
# Service account that the Cloud Run service runs as
# ----------------------------------------------------------------------
resource "google_service_account" "run" {
  account_id   = "football-api"
  display_name = "Football API Cloud Run runtime"
}

resource "google_storage_bucket_iam_member" "run_bucket_access" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.run.email}"
}

# ----------------------------------------------------------------------
# Secrets — odds API key and shared API secret
# ----------------------------------------------------------------------
resource "google_secret_manager_secret" "odds_api_key" {
  secret_id = "odds-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "api_key" {
  secret_id = "football-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "betfair_app_key" {
  secret_id = "betfair-app-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "betfair_username" {
  secret_id = "betfair-username"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "betfair_password" {
  secret_id = "betfair-password"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "betfair_cert" {
  secret_id = "betfair-cert"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "betfair_key" {
  secret_id = "betfair-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "run_odds" {
  secret_id = google_secret_manager_secret.odds_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_api" {
  secret_id = google_secret_manager_secret.api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_betfair_app_key" {
  secret_id = google_secret_manager_secret.betfair_app_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_betfair_username" {
  secret_id = google_secret_manager_secret.betfair_username.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_betfair_password" {
  secret_id = google_secret_manager_secret.betfair_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_betfair_cert" {
  secret_id = google_secret_manager_secret.betfair_cert.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_betfair_key" {
  secret_id = google_secret_manager_secret.betfair_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

# ----------------------------------------------------------------------
# Cloud Run service
# ----------------------------------------------------------------------
resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  # The image is owned by the GitHub Actions deploy pipeline after the first
  # successful build. Terraform creates the service with a placeholder so the
  # resource can exist before any image is built; CI overwrites the image on
  # every push without Terraform reverting it on the next `apply`.
  lifecycle {
    ignore_changes = [
      client,
      client_version,
      template[0].containers[0].image,
    ]
  }

  template {
    service_account = google_service_account.run.email

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello"

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "GCS_BUCKET"
        value = google_storage_bucket.data.name
      }
      env {
        name  = "MODEL_BLOB"
        value = "models/pl_model.pkl"
      }
      env {
        name  = "ARCHIVE_PREFIX"
        value = "predictions/"
      }
      env {
        name = "ODDS_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.odds_api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "BETFAIR_APP_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.betfair_app_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "BETFAIR_USERNAME"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.betfair_username.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "BETFAIR_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.betfair_password.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "BETFAIR_CERT"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.betfair_cert.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "BETFAIR_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.betfair_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ----------------------------------------------------------------------
# Cloud Scheduler — hits /predictions/archive daily
# ----------------------------------------------------------------------
resource "google_service_account" "scheduler" {
  account_id   = "football-scheduler"
  display_name = "Football daily scheduler"
}

resource "google_cloud_scheduler_job" "daily_archive" {
  name      = "football-daily-archive"
  schedule  = var.schedule_cron
  time_zone = var.schedule_tz
  region    = var.region

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.api.uri}/predictions/archive?limit=20"
    headers = {
      "X-API-Key"    = "__SET_VIA_GCLOUD__" # populate manually after `terraform apply` (or use Secret Manager + gcloud)
      "Content-Type" = "application/json"
    }
  }

  depends_on = [google_project_service.apis]
}

# ----------------------------------------------------------------------
# Workload Identity Federation — lets GitHub Actions deploy without keys
# ----------------------------------------------------------------------
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions pool"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }

  attribute_condition = "assertion.repository == \"${var.github_repo}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "deployer" {
  account_id   = "github-deployer"
  display_name = "GitHub Actions deployer"
}

resource "google_project_iam_member" "deployer_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_project_iam_member" "deployer_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_actas_run" {
  service_account_id = google_service_account.run.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_wif" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}
