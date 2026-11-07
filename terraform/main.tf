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

# Scorers (scorersonline.uk) bot API keys - one per account, so the model and
# the Betfair market each play the league as their own player. Values are set
# out of band once the two bot accounts exist; the API returns 503 for a stream
# whose key is still empty, so the infra can land ahead of the accounts.
resource "google_secret_manager_secret" "scorers_model_api_key" {
  secret_id = "scorers-model-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "scorers_betfair_api_key" {
  secret_id = "scorers-betfair-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

# Cloud Run resolves `versions/latest` at deploy time and refuses the revision
# if the secret has no versions at all, so a secret created ahead of its real
# value would break every deploy until the account existed. Seed a placeholder
# version instead, and ignore later changes so adding the real key out of band
# isn't reverted on the next apply. The app treats this exact value as "not
# configured" and returns 503 for that stream - loudly unconfigured rather than
# silently submitting with a junk key.
resource "google_secret_manager_secret_version" "scorers_model_api_key_placeholder" {
  secret      = google_secret_manager_secret.scorers_model_api_key.id
  secret_data = local.scorers_key_placeholder

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_secret_manager_secret_version" "scorers_betfair_api_key_placeholder" {
  secret      = google_secret_manager_secret.scorers_betfair_api_key.id
  secret_data = local.scorers_key_placeholder

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_secret_manager_secret_iam_member" "run_scorers_model" {
  secret_id = google_secret_manager_secret.scorers_model_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "run_scorers_betfair" {
  secret_id = google_secret_manager_secret.scorers_betfair_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
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
      # The scheduler jobs authenticate with OIDC tokens rather than a shared
      # secret. The service is public (the HTML views are bookmarked on a
      # phone), so Cloud Run can't enforce this for us - the app verifies the
      # token against this service account itself.
      env {
        name  = "OIDC_SERVICE_ACCOUNT"
        value = google_service_account.scheduler.email
      }
      env {
        name  = "OIDC_AUDIENCE"
        value = local.service_audience
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
        name = "SCORERS_MODEL_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.scorers_model_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "SCORERS_BETFAIR_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.scorers_betfair_api_key.secret_id
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

  # The secret versions, not just the secrets: Cloud Run resolves `latest` when
  # it creates the revision and fails if the secret has no version yet.
  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.scorers_model_api_key_placeholder,
    google_secret_manager_secret_version.scorers_betfair_api_key_placeholder,
  ]
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

locals {
  # Must match SCORERS_KEY_PLACEHOLDER in app.py.
  scorers_key_placeholder = "__UNSET__"

  # Audience the scheduler mints its OIDC tokens for, and that the app verifies.
  # A fixed string rather than the service URL: the service needs this value as
  # an env var, and deriving it from the service would be a dependency cycle.
  # If unauthenticated access is ever removed, Cloud Run will require the
  # audience to be the real service URL and this must change to match.
  service_audience = "https://${var.service_name}"
}

# Checks football-data.co.uk for newly published results and retrains only if
# something changed. Runs before the archive job so the day's archived
# predictions come from a model trained on the freshest available data.
resource "google_cloud_scheduler_job" "daily_retrain" {
  name      = "football-daily-retrain"
  schedule  = var.retrain_cron
  time_zone = var.schedule_tz
  region    = var.region

  # Training is a single long request; let it finish rather than retrying into
  # a 409 from the in-progress lock.
  attempt_deadline = "600s"
  retry_config {
    retry_count = 0
  }

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.api.uri}/model/refresh"
    headers = {
      "Content-Type" = "application/json"
    }
    oidc_token {
      service_account_email = google_service_account.scheduler.email
      audience              = local.service_audience
    }
  }

  depends_on = [google_project_service.apis]
}

# One job per stream rather than one job submitting both: a Betfair outage
# must not stop the model playing, and the two rows on the leaderboard should
# fail independently.
resource "google_cloud_scheduler_job" "daily_submit" {
  for_each = toset(["model", "betfair"])

  name      = "football-daily-submit-${each.key}"
  schedule  = var.submit_cron
  time_zone = var.schedule_tz
  region    = var.region

  attempt_deadline = "300s"
  # Submissions are idempotent (Scorers overwrites on repeat POSTs), so a retry
  # is safe and worth having - a missed deadline means no picks that round.
  retry_config {
    retry_count = 2
  }

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.api.uri}/submissions/scorers?stream=${each.key}&limit=20"
    headers = {
      "Content-Type" = "application/json"
    }
    oidc_token {
      service_account_email = google_service_account.scheduler.email
      audience              = local.service_audience
    }
  }

  depends_on = [google_project_service.apis]
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
      "Content-Type" = "application/json"
    }
    oidc_token {
      service_account_email = google_service_account.scheduler.email
      audience              = local.service_audience
    }
  }

  depends_on = [google_project_service.apis]
}

# Not strictly required while the service allows unauthenticated access, but
# keeps the scheduler working if that's ever tightened.
resource "google_cloud_run_v2_service_iam_member" "scheduler_invoker" {
  name     = google_cloud_run_v2_service.api.name
  location = google_cloud_run_v2_service.api.location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
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
