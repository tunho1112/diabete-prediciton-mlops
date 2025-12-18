# Ref: https://github.com/terraform-google-modules/terraform-google-kubernetes-engine/blob/master/examples/simple_autopilot_public
# To define that we will use GCP
terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.80.0" // Provider version
    }
  }
  required_version = "1.12.2" // Terraform version
}


provider "google" {
  project     = var.project_id
  region      = var.region
}

// Google Kubernetes Engine
resource "google_container_cluster" "primary" {
  name     = "${var.project_id}-gke-k5"
  location = var.zone

  // Using standard cluster instead of Autopilot to fit within CPU quota
  // Remove default node pool as we'll create a custom one
  remove_default_node_pool = true
  initial_node_count       = 1
}

// Custom node pool with smaller instances to fit quota
resource "google_container_node_pool" "primary_nodes" {
  name       = "${var.project_id}-node-pool-1"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = 1 // Start with 1 node (2 CPUs)

  node_config {
    machine_type = "e2-small" // 2 vCPUs per node
    disk_size_gb = 20
    disk_type    = "pd-standard"
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 2 // Max 2 nodes (4 CPUs total) to stay within quota
  }
}

resource "google_compute_instance" "jenkins_mle" {
  name         = "jenkins-mle"
  machine_type = var.machine_type
  zone         = var.zone
  project      = var.project_id

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-standard"
    }
    auto_delete = true
  }

  network_interface {
    network = "default"
    access_config {
      // This enables external IP
    }
  }
}

resource "google_compute_firewall" "jenkins_ports" {
  name    = "allow-jenkins-ports"
  network = "default"
  allow {
    protocol = "tcp"
    ports    = ["8081", "50000"]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["jenkins"]
  direction     = "INGRESS"
  description   = "Allow inbound TCP traffic for Jenkins on ports 8081 and 50000"
}
