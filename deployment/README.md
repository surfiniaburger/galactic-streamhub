# Migration Guide: Cloud Run to Google Kubernetes Engine (GKE) Autopilot for "Galatic Streamhub"

This document details the steps taken to migrate the "Galatic Streamhub" application from Google Cloud Run to Google Kubernetes Engine (GKE) Autopilot. The migration focuses on setting up a scalable and secure environment on GKE, including HTTPS, WebSocket support, proper secret management, and automated scaling.

**Original State:** Application running on Cloud Run, defined by a Knative Serving YAML.
**Target State:** Application running on GKE Autopilot, exposed via HTTPS with a custom domain, using Kubernetes-native configurations for deployments, services, scaling, and secrets.

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Phase 1: Understanding the Cloud Run Configuration](#phase-1-understanding-the-cloud-run-configuration)
3.  [Phase 2: Setting up the GKE Autopilot Cluster](#phase-2-setting-up-the-gke-autopilot-cluster)
    *   [2.1 Create a GKE Node Service Account (IAM)](#21-create-a-gke-node-service-account-iam)
    *   [2.2 Create the Autopilot Cluster](#22-create-the-autopilot-cluster)
    *   [2.3 Connect to the Cluster](#23-connect-to-the-cluster)
4.  [Phase 3: Application Code Modifications](#phase-3-application-code-modifications)
    *   [3.1 Standardize Secret Handling (Environment Variables)](#31-standardize-secret-handling-environment-variables)
    *   [3.2 Rebuild and Push Docker Image](#32-rebuild-and-push-docker-image)
5.  [Phase 4: Kubernetes Resource Configuration](#phase-4-kubernetes-resource-configuration)
    *   [4.1 Kubernetes Secrets (`kubernetes-secrets.yaml`)](#41-kubernetes-secrets-kubernetes-secretsyaml)
    *   [4.2 Workload Identity Setup](#42-workload-identity-setup)
    *   [4.3 Deployment (`kubernetes-deployment.yaml`)](#43-deployment-kubernetes-deploymentyaml)
    *   [4.4 Horizontal Pod Autoscaler (HPA) (`kubernetes-hpa.yaml`)](#44-horizontal-pod-autoscaler-hpa-kubernetes-hpayaml)
6.  [Phase 5: Setting up Domain, HTTPS, and External Access (Ingress)](#phase-5-setting-up-domain-https-and-external-access-ingress)
    *   [5.1 Acquire Domain (`galactic-streamhub.com`)](#51-acquire-domain-galactic-streamhubcom)
    *   [5.2 Reserve Static IP for Ingress](#52-reserve-static-ip-for-ingress)
    *   [5.3 Create ManagedCertificate (`managed-certificate.yaml`)](#53-create-managedcertificate-managed-certificateyaml)
    *   [5.4 Create BackendConfig (`backend-config.yaml`)](#54-create-backendconfig-backend-configyaml)
    *   [5.5 Update Service (`kubernetes-service.yaml`) for Ingress](#55-update-service-kubernetes-serviceyaml-for-ingress)
    *   [5.6 Create Ingress (`ingress.yaml`)](#56-create-ingress-ingressyaml)
    *   [5.7 Configure DNS (Cloud DNS)](#57-configure-dns-cloud-dns)
7.  [Phase 6: Deployment and Verification](#phase-6-deployment-and-verification)
    *   [6.1 Apply Kubernetes Manifests](#61-apply-kubernetes-manifests)
    *   [6.2 Update Client Application and Firebase](#62-update-client-application-and-firebase)
    *   [6.3 Monitor and Test](#63-monitor-and-test)
8.  [Troubleshooting and Key Learnings](#troubleshooting-and-key-learnings)
9.  [Final Kubernetes Manifests Summary](#final-kubernetes-manifests-summary)

---

## Prerequisites

*   Google Cloud SDK (`gcloud`) installed and authenticated.
*   `kubectl` command-line tool installed.
*   A Google Cloud Project with billing enabled.
*   Necessary IAM permissions (e.g., Kubernetes Engine Admin, Service Account Admin, DNS Administrator, Compute Network Admin).
*   Docker installed locally for building images.
*   Access to the application's source code (`main.py`, `agent_config.py`, etc.).
*   Container registry (Artifact Registry preferred) for storing Docker images.
*   Existing Firebase project setup (for Auth and App Check).

---

## Phase 1: Understanding the Cloud Run Configuration

First, we obtained the existing Cloud Run service definition to understand its configuration, especially environment variables, secrets, image, and scaling parameters.

**Command to download Cloud Run YAML:**
```bash
gcloud run services describe galatic-streamhub \
    --format export \
    --region us-central1 \
    --project silver-455021 > galatic-streamhub-cloudrun-initial.yaml
```
*(Replace `silver-455021` with your Project ID and `us-central1` with your region if different. The output file was, for example, `galatic-streamhub-cloudrun-initial.yaml`)*

**Key details from an example Cloud Run YAML (e.g., the last one provided):**
*   **Image:** `us-central1-docker.pkg.dev/silver-455021/cloud-run-source-deploy/galatic-streamhub@sha256:2509233433e1127...`
*   **Environment Variables:** `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `mongodb-uri`, `MULTIMODAL_MONGODB_URI`, `GOOGLE_MAPS_API_KEY`, `FIREBASE_PROJECT_ID`.
*   **Service Account:** `galatic-streamhub@silver-455021.iam.gserviceaccount.com`
*   **Scaling:** `autoscaling.knative.dev/maxScale: '21'`
*   **Resources:** CPU 1000m, Memory 4G.

---

## Phase 2: Setting up the GKE Autopilot Cluster

GKE Autopilot was chosen to reduce operational overhead for managing the cluster's nodes.

### 2.1 Create a GKE Node Service Account (IAM)

A dedicated, minimally-privileged IAM service account was created for GKE nodes instead of using the Compute Engine default service account.

**Commands:**
1.  **Create Service Account:**
    ```bash
    gcloud iam service-accounts create gke-autopilot-node-sa \
        --project=silver-455021 \
        --display-name="GKE Autopilot Node Service Account for Galatic Streamhub"
    ```
    *(This creates `gke-autopilot-node-sa@silver-455021.iam.gserviceaccount.com`)*

2.  **Grant Necessary Roles:**
    ```bash
    # For logs and metrics
    gcloud projects add-iam-policy-binding silver-455021 --member="serviceAccount:gke-autopilot-node-sa@silver-455021.iam.gserviceaccount.com" --role="roles/monitoring.metricWriter"
    gcloud projects add-iam-policy-binding silver-455021 --member="serviceAccount:gke-autopilot-node-sa@silver-455021.iam.gserviceaccount.com" --role="roles/logging.logWriter"
    gcloud projects add-iam-policy-binding silver-455021 --member="serviceAccount:gke-autopilot-node-sa@silver-455021.iam.gserviceaccount.com" --role="roles/stackdriver.resourceMetadata.writer" # Or ensure monitoring.viewer is present

    # For pulling images from Artifact Registry/GCR
    gcloud projects add-iam-policy-binding silver-455021 --member="serviceAccount:gke-autopilot-node-sa@silver-455021.iam.gserviceaccount.com" --role="roles/storage.objectViewer"
    gcloud projects add-iam-policy-binding silver-455021 --member="serviceAccount:gke-autopilot-node-sa@silver-455021.iam.gserviceaccount.com" --role="roles/artifactregistry.reader"
    ```

### 2.2 Create the Autopilot Cluster

```bash
gcloud container clusters create-auto galactic-streamhub-cluster \
    --project=silver-455021 \
    --region=us-central1 \
    --service-account="gke-autopilot-node-sa@silver-455021.iam.gserviceaccount.com" \
    --release-channel=regular # Or your preferred release channel
```

### 2.3 Connect to the Cluster

```bash
gcloud container clusters get-credentials galactic-streamhub-cluster \
    --region us-central1 \
    --project silver-455021
```

---

## Phase 3: Application Code Modifications

The application code (`main.py`) was updated to consistently read all sensitive configurations (including `GOOGLE_MAPS_API_KEY`) from environment variables. Previously, the Maps API key was handled differently (potentially via volume mount in Cloud Run and Secret Manager fetching in an intermediate `main.py` version).

### 3.1 Standardize Secret Handling (Environment Variables)

*   Removed direct Secret Manager fetching logic for `GOOGLE_MAPS_API_KEY` from `main.py`.
*   Ensured `main.py` uses `os.environ.get("GOOGLE_MAPS_API_KEY")` and `os.environ.get("FIREBASE_PROJECT_ID")`.

**Example snippet from `main.py`'s `app_lifespan` after modification:**
```python
# In app_lifespan
# ...
google_maps_api_key_from_env = os.environ.get("GOOGLE_MAPS_API_KEY")
firebase_project_id_from_env = os.environ.get("FIREBASE_PROJECT_ID") # Or directly use os.environ.get in Firebase init

if google_maps_api_key_from_env:
    maps_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-google-maps"],
        env={"GOOGLE_MAPS_API_KEY": google_maps_api_key_from_env}
    )
    current_server_configs["maps"] = maps_server_params
# ...
```

### 3.2 Rebuild and Push Docker Image

After code modifications, the Docker image was rebuilt and pushed to Artifact Registry.
```bash
# Example Docker build and push commands
docker build -t us-central1-docker.pkg.dev/silver-455021/cloud-run-source-deploy/galatic-streamhub:new-gke-tag .
docker push us-central1-docker.pkg.dev/silver-455021/cloud-run-source-deploy/galatic-streamhub:new-gke-tag

# For specific SHA usage, get the SHA after pushing, e.g., from Artifact Registry UI or:
# gcloud artifacts docker images describe \
#    us-central1-docker.pkg.dev/silver-455021/cloud-run-source-deploy/galatic-streamhub:new-gke-tag \
#    --format='get(image_summary.digest)'
# The last Cloud Run YAML used: sha256:2509233433e1127ca16dc5a2b5ef278b06cb3bed1c9651737cfa4eeef3135cc2
```
The image used in the final deployment YAML was: `us-central1-docker.pkg.dev/silver-455021/cloud-run-source-deploy/galatic-streamhub@sha256:2509233433e1127ca16dc5a2b5ef278b06cb3bed1c9651737cfa4eeef3135cc2`

---

## Phase 4: Kubernetes Resource Configuration

All Kubernetes resources were deployed to the `default` namespace for simplicity. For production, a dedicated namespace (e.g., `galatic-streamhub-ns`) is recommended.

### 4.1 Kubernetes Secrets (`kubernetes-secrets.yaml`)

Created Kubernetes `Secret` resources to store sensitive data. Values were base64 encoded.
**File: `kubernetes-secrets.yaml`**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: google-api-credentials
  namespace: default
type: Opaque
data:
  apiKey: # BASE64_ENCODED_VALUE_OF_AIz..yA_p8iR...
---
apiVersion: v1
kind: Secret
metadata:
  name: mongodb-credentials
  namespace: default
type: Opaque
data:
  uri: # BASE64_ENCODED_VALUE_OF_mongodb+srv://jdmasciano2:S...ha@cluster0...
---
apiVersion: v1
kind: Secret
metadata:
  name: multimodal-mongodb-credentials
  namespace: default
type: Opaque
data:
  uri: # BASE64_ENCODED_VALUE_OF_mongodb+srv://jdmasciano2:Fx...@cluster0...
---
apiVersion: v1
kind: Secret
metadata:
  name: google-maps-api-key-k8s
  namespace: default
type: Opaque
data:
  mapsApiKey: # BASE64_ENCODED_VALUE_OF_AIzaSyD...
```
**Apply Command:**
```bash
kubectl apply -f kubernetes-secrets.yaml -n default
```

### 4.2 Workload Identity Setup

Configured Workload Identity to allow Kubernetes Service Accounts to impersonate Google Cloud IAM Service Accounts.
The application uses `galatic-streamhub@silver-455021.iam.gserviceaccount.com` (Application SA).

**Commands:**
1.  **Create Kubernetes Service Account (KSA):**
    ```bash
    kubectl create serviceaccount galatic-streamhub-ksa --namespace default
    ```

2.  **Allow KSA to Impersonate Application IAM SA:**
    ```bash
    gcloud iam service-accounts add-iam-policy-binding "galatic-streamhub@silver-455021.iam.gserviceaccount.com" \
        --role="roles/iam.workloadIdentityUser" \
        --member="serviceAccount:silver-455021.svc.id.goog[default/galatic-streamhub-ksa]" \
        --project=silver-455021
    ```

3.  **Annotate KSA:**
    ```bash
    kubectl annotate serviceaccount galatic-streamhub-ksa \
        --namespace default \
        iam.gke.io/gcp-service-account="galatic-streamhub@silver-455021.iam.gserviceaccount.com"
    ```
    *Ensure `galatic-streamhub@silver-455021.iam.gserviceaccount.com` has necessary permissions (e.g., `roles/aiplatform.user`).*

### 4.3 Deployment (`kubernetes-deployment.yaml`)

Defines the application's pods, image, environment variables (sourcing secrets), resource requests/limits, and probes.
**File: `kubernetes-deployment.yaml`** (final version)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: galatic-streamhub
  namespace: default
  labels:
    app: galatic-streamhub
spec:
  replicas: 1
  selector:
    matchLabels:
      app: galatic-streamhub
  template:
    metadata:
      labels:
        app: galatic-streamhub
    spec:
      serviceAccountName: galatic-streamhub-ksa
      containers:
      - name: galatic-streamhub-1
        image: us-central1-docker.pkg.dev/silver-455021/cloud-run-source-deploy/galatic-streamhub@sha256:2509233433e1127ca16dc5a2b5ef278b06cb3bed1c9651737cfa4eeef3135cc2
        ports:
        - containerPort: 8080
          name: http1
        env:
        - name: GOOGLE_GENAI_USE_VERTEXAI
          value: '1'
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: google-api-credentials
              key: apiKey
        - name: GOOGLE_CLOUD_PROJECT
          value: silver-455021
        - name: GOOGLE_CLOUD_LOCATION
          value: us-central1
        - name: mongodb-uri
          valueFrom:
            secretKeyRef:
              name: mongodb-credentials
              key: uri
        - name: MULTIMODAL_MONGODB_URI
          valueFrom:
            secretKeyRef:
              name: multimodal-mongodb-credentials
              key: uri
        - name: GOOGLE_MAPS_API_KEY
          valueFrom:
            secretKeyRef:
              name: google-maps-api-key-k8s
              key: mapsApiKey
        - name: FIREBASE_PROJECT_ID
          value: 'studio-l13dd'
        resources:
          limits:
            cpu: "1000m"
            memory: "4G"
          requests:
            cpu: "500m"
            memory: "2G"
        startupProbe:
          tcpSocket:
            port: 8080
          failureThreshold: 1
          periodSeconds: 240
          timeoutSeconds: 240
```

### 4.4 Horizontal Pod Autoscaler (HPA) (`kubernetes-hpa.yaml`)

Manages automatic scaling of pods based on CPU utilization.
**File: `kubernetes-hpa.yaml`**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: galatic-streamhub-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: galatic-streamhub
  minReplicas: 1
  maxReplicas: 21
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Phase 5: Setting up Domain, HTTPS, and External Access (Ingress)

The goal was to expose the application via `https://app.galactic-streamhub.com`.

### 5.1 Acquire Domain (`galactic-streamhub.com`)

Domain `galactic-streamhub.com` was registered using Google Cloud Domains. Cloud DNS was used for hosting.

### 5.2 Reserve Static IP for Ingress

```bash
gcloud compute addresses create galactic-streamhub-static-ip --global --project=silver-455021
# Note down the IP, e.g., XX.YY.ZZ.AA
gcloud compute addresses describe galactic-streamhub-static-ip --global --project=silver-455021 --format="value(address)"
```

### 5.3 Create ManagedCertificate (`managed-certificate.yaml`)

For Google-managed SSL.
**File: `managed-certificate.yaml`**
```yaml
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: galactic-streamhub-certificate
  namespace: default
spec:
  domains:
    - app.galactic-streamhub.com
```

### 5.4 Create BackendConfig (`backend-config.yaml`)

To configure longer timeouts for WebSocket connections.
**File: `backend-config.yaml`**
```yaml
apiVersion: cloud.google.com/v1
kind: BackendConfig
metadata:
  name: galactic-streamhub-bconfig
  namespace: default
spec:
  timeoutSec: 86400 # 24 hours
```

### 5.5 Update Service (`kubernetes-service.yaml`) for Ingress

The Service type was changed to `NodePort` and annotated for Ingress and BackendConfig.
**File: `kubernetes-service.yaml`** (final version)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: galatic-streamhub-svc
  namespace: default
  labels:
    app: galatic-streamhub
  annotations:
    cloud.google.com/neg: '{"ingress": true}'
    cloud.google.com/backend-config: '{"default": "galactic-streamhub-bconfig"}'
spec:
  type: NodePort
  selector:
    app: galatic-streamhub
  ports:
  - name: http1
    protocol: TCP
    port: 80
    targetPort: 8080 # Or 'http1' if that's the name in Deployment's container port
```

### 5.6 Create Ingress (`ingress.yaml`)

To manage external L7 load balancing and SSL termination.
**File: `ingress.yaml`**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: galactic-streamhub-ingress
  namespace: default
  annotations:
    kubernetes.io/ingress.global-static-ip-name: "galactic-streamhub-static-ip"
    networking.gke.io/managed-certificates: "galactic-streamhub-certificate"
spec:
  rules:
  - host: app.galactic-streamhub.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: galatic-streamhub-svc
            port:
              name: http1 # Or number: 80```

### 5.7 Configure DNS (Cloud DNS)

An `A` record was created in Cloud DNS for `app.galactic-streamhub.com` pointing to the reserved static IP (`XX.YY.ZZ.AA`).
1.  Go to Cloud DNS in GCP Console.
2.  Select zone for `galactic-streamhub.com`.
3.  Add Record Set:
    *   **DNS Name:** `app`
    *   **Type:** `A`
    *   **IPv4 Address:** `XX.YY.ZZ.AA` (the static IP)
    *   **TTL:** e.g., `300`

---

## Phase 6: Deployment and Verification

### 6.1 Apply Kubernetes Manifests

The order of application can matter for dependencies (e.g., ServiceAccount before Deployment, Service before Ingress refers to it).
```bash
# 1. Secrets (if not already applied or changed)
kubectl apply -f kubernetes-secrets.yaml -n default

# 2. ServiceAccount (KSA) is created via gcloud/kubectl create sa

# 3. BackendConfig
kubectl apply -f backend-config.yaml -n default

# 4. Service (updated for NodePort and BackendConfig)
kubectl apply -f kubernetes-service.yaml -n default

# 5. Deployment (with latest image and env vars)
kubectl apply -f kubernetes-deployment.yaml -n default

# 6. HPA
kubectl apply -f kubernetes-hpa.yaml -n default

# 7. ManagedCertificate
kubectl apply -f managed-certificate.yaml -n default

# 8. Ingress
kubectl apply -f ingress.yaml -n default
```

### 6.2 Update Client Application and Firebase

1.  **Client-side `app.js`:** Updated WebSocket URL to `wss://app.galactic-streamhub.com/ws`.
2.  **Firebase App Check:** Added `app.galactic-streamhub.com` to allowed domains for the web app in Firebase Console.
3.  **Firebase Authentication:** Added `app.galactic-streamhub.com` to authorized domains.

### 6.3 Monitor and Test

*   **Rollout Status:**
    ```bash
    kubectl rollout status deployment/galatic-streamhub -n default
    ```
*   **Pod Logs:**
    ```bash
    kubectl logs deployment/galatic-streamhub -n default -f --tail=200
    ```
*   **Ingress and Certificate Status:**
    ```bash
    kubectl get managedcertificate galactic-streamhub-certificate -n default -w
    kubectl describe managedcertificate galactic-streamhub-certificate -n default
    kubectl get ingress galactic-streamhub-ingress -n default -w
    kubectl describe ingress galactic-streamhub-ingress -n default
    ```
*   **Access Application:** `https://app.galactic-streamhub.com`
*   **Test Functionality:** Thoroughly tested text, audio, and video WebSocket communication.

---

## Troubleshooting and Key Learnings

*   **HTTP vs. HTTPS (ws:// vs. wss://):** Initial issues with audio/video and Firebase App Check on GKE were due to the default `type: LoadBalancer` Service exposing HTTP. Moving to GKE Ingress with `ManagedCertificate` resolved this by providing `wss://`.
*   **Firebase App Check:** Requires the correct (HTTPS) domain to be registered. "ReCAPTCHA error" was a key indicator.
*   **AudioWorklet Security:** Browser features like `AudioWorklet` often require a secure context (HTTPS).
*   **WebSocket Timeouts:** Long-lived WebSockets require configuring backend timeouts on the L7 Load Balancer using `BackendConfig` (`timeoutSec`).
*   **MCP Toolset Shutdown (`RuntimeError: Attempted to exit cancel scope...`):**
    *   This error appeared during application shutdown on GKE.
    *   The root cause was identified as a potential mismatch in how `MCPToolset` instances were managed and closed, particularly concerning the `async_generator object stdio_client`.
    *   **Resolution involved:** Reviewing ADK documentation for `MCPToolset` lifecycle management, ensuring `async with` or the returned exit stacks from `MCPToolset.from_server` are used correctly for cleanup, or simplifying the toolset closing logic in `app_lifespan`. (The exact fix here depends on the deep specifics of ADK's intended usage, which might involve using the `exit_stack` from `_collect_tools_stack` or ensuring `toolset.close()` is fully compatible with `anyio`'s task scopes when the toolset is manually instantiated).
*   **Namespace Consistency:** All related Kubernetes resources should be in the same namespace.
*   **Resource Requests for Autopilot:** `spec.template.spec.containers.resources.requests` (CPU and memory) are mandatory for GKE Autopilot deployments.

---

## Final Kubernetes Manifests Summary

*   `kubernetes-secrets.yaml` (for sensitive data)
*   `kubernetes-deployment.yaml` (application pods, image, env vars)
*   `kubernetes-service.yaml` (NodePort, NEG & BackendConfig annotations)
*   `kubernetes-hpa.yaml` (autoscaling definition)
*   `managed-certificate.yaml` (for SSL)
*   `backend-config.yaml` (for LB timeouts)
*   `ingress.yaml` (routing, SSL termination, static IP)

This migration provides "Galatic Streamhub" with a more robust, scalable, and configurable platform on GKE Autopilot, leveraging Kubernetes-native features for a production-grade deployment.
```
