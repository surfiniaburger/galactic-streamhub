# Set variables for clarity
PROJECT_ID="silver-455021"
REGION="us-central1"
TEMPLATE_ID="my-curl-generated-template" # Using a different ID to avoid conflicts
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Execute the curl command to create the template
curl -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "filter_config": {
      "piAndJailbreakFilterSettings": {
        "filterEnforcement": "ENABLED"
      },
      "maliciousUriFilterSettings": {
        "filterEnforcement": "ENABLED"
      },
      "raiSettings": {
        "raiFilters": [
          {
            "filterType": "SEXUALLY_EXPLICIT",
            "confidenceLevel": "LOW_AND_ABOVE"
          },
          {
            "filterType": "HATE_SPEECH",
            "confidenceLevel": "LOW_AND_ABOVE"
          },
          {
            "filterType": "HARASSMENT",
            "confidenceLevel": "LOW_AND_ABOVE"
          },
          {
            "filterType": "DANGEROUS",
            "confidenceLevel": "LOW_AND_ABOVE"
          }
        ]
      },
      "sdpSettings": {
        "basicConfig": {
          "filterEnforcement": "ENABLED"
        }
      }
    }
  }' \
  "https://modelarmor.$REGION.rep.googleapis.com/v1alpha/projects/$PROJECT_ID/locations/$REGION/templates?template_id=$TEMPLATE_ID"