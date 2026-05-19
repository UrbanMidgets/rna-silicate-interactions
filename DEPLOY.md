# Deployment to Google Cloud Run

This Streamlit application can be hosted on Google Cloud Run for free (within the free tier).

## Prerequisites

1.  A Google Cloud account.
2.  [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured.
3.  Docker (optional, if deploying via source).

## Steps to Deploy

1.  **Initialize Google Cloud:**
    ```bash
    gcloud init
    ```

2.  **Deploy from Source:**
    Run the following command from the root of the repository:
    ```bash
    gcloud run deploy rna-interactions --source . --region us-central1 --allow-unauthenticated
    ```
    *Replace `us-central1` with your preferred region.*

3.  **Wait for deployment:**
    Once finished, it will provide a URL (e.g., `https://rna-interactions-xyz.a.run.app`).

## Custom Domain Mapping

If you have a domain, you can map it to your Cloud Run service:

1.  Go to the [Cloud Run Console](https://console.cloud.google.com/run).
2.  Click on **Manage Custom Domains**.
3.  Click **Add Mapping**.
4.  Select your service (`rna-interactions`) and enter your domain.
5.  Follow the instructions to verify ownership and update your DNS records (CNAME/A records).

## Free Tier Notes

Google Cloud Run has a generous free tier:
- First 180,000 vCPU-seconds per month are free.
- First 360,000 GiB-seconds per month are free.
- 2 million requests per month are free.

This should be plenty for a research application with moderate traffic.
