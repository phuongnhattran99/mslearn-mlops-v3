"""
Deploy an MLflow model to a Managed Online Endpoint in Azure Machine Learning.

Usage:
    python src/deploy_to_online_endpoint.py \
        --subscription_id  <sub>  \
        --resource_group   <rg>   \
        --workspace_name   <ws>   \
        --endpoint_name    diabetes-endpoint \
        --deployment_name  blue
"""
import argparse
import logging
import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# 1. Connect to workspace
# ──────────────────────────────────────────────────────────────────
def get_ml_client(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(credential, subscription_id, resource_group, workspace_name)


# ──────────────────────────────────────────────────────────────────
# 2. Ensure endpoint exists (create if not)
# ──────────────────────────────────────────────────────────────────
def create_or_update_endpoint(
    ml_client: MLClient,
    endpoint_name: str,
) -> ManagedOnlineEndpoint:
    """Create the endpoint if it does not exist; return the endpoint object."""
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        logger.info(f"Endpoint '{endpoint_name}' already exists — reusing.")
    except Exception:
        logger.info(f"Endpoint '{endpoint_name}' not found — creating ...")
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Diabetes prediction real-time endpoint",
            auth_mode="key",
            tags={"project": "mslearn-mlops"},
        )
        endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Endpoint '{endpoint_name}' created.")

    return endpoint


# ──────────────────────────────────────────────────────────────────
# 3. Register MLflow model + create / update deployment
# ──────────────────────────────────────────────────────────────────
def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
) -> ManagedOnlineDeployment:
    """
    Register the MLflow model from the local ./model folder, then
    create (or update) a managed online deployment.

    Key design decision:
        For AssetTypes.MLFLOW_MODEL, Azure ML auto-infers the scoring
        script and the inference environment directly from the model
        artifacts (MLmodel + conda.yaml).  Passing an explicit
        `environment=` or `code_configuration=` would fight with that
        auto-inference and is NOT needed here.
    """

    # ── 3a. Register model ─────────────────────────────────────────
    logger.info("Registering model from './model' ...")
    registered_model = ml_client.models.create_or_update(
        Model(
            name="diabetes-model",
            path="./model",                    # repo-root model/ folder
            type=AssetTypes.MLFLOW_MODEL,      # tells AML this is MLflow
            description="Diabetes classification MLflow model",
        )
    )
    logger.info(
        f"Registered model: '{registered_model.name}' v{registered_model.version}"
    )

    # ── 3b. Define deployment ──────────────────────────────────────
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=registered_model,
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )

    logger.info(
        f"Creating / updating deployment '{deployment_name}' "
        f"on endpoint '{endpoint_name}' ..."
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # ── 3c. Route 100 % of traffic to this deployment ─────────────
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    logger.info(f"Traffic set: 100 % → '{deployment_name}'")

    return ml_client.online_deployments.get(
        name=deployment_name,
        endpoint_name=endpoint_name,
    )


# ──────────────────────────────────────────────────────────────────
# 4. Write GitHub Actions output (FIX #5: replace deprecated ::set-output)
# ──────────────────────────────────────────────────────────────────
def set_github_output(name: str, value: str) -> None:
    """Write a step output using the modern $GITHUB_OUTPUT file API."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}={value}\n")
    else:
        # Running locally — just print
        print(f"[output] {name}={value}")


# ──────────────────────────────────────────────────────────────────
# 5. Entry point
# ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy MLflow model to Azure ML Managed Online Endpoint"
    )
    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group",  required=True)
    parser.add_argument("--workspace_name",  required=True)
    parser.add_argument("--endpoint_name",   default="diabetes-endpoint")
    parser.add_argument("--deployment_name", default="blue")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Connecting to Azure ML workspace ...")
    ml_client = get_ml_client(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )
    logger.info(
        f"Connected → sub={args.subscription_id} "
        f"| rg={args.resource_group} | ws={args.workspace_name}"
    )

    # ── Create / reuse endpoint ────────────────────────────────────
    logger.info(f"Ensuring endpoint '{args.endpoint_name}' exists ...")
    endpoint = create_or_update_endpoint(ml_client, args.endpoint_name)

    # ── Deploy model ───────────────────────────────────────────────
    logger.info(f"Creating or updating deployment '{args.deployment_name}' ...")
    deployment = create_or_update_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint.name,
        deployment_name=args.deployment_name,
    )

    # ── Print results ──────────────────────────────────────────────
    scoring_uri = ml_client.online_endpoints.get(args.endpoint_name).scoring_uri
    logger.info(f"Deployment state : {deployment.provisioning_state}")
    logger.info(f"Scoring URI      : {scoring_uri}")

    # FIX #5: use $GITHUB_OUTPUT instead of deprecated ::set-output
    set_github_output("scoring_uri", scoring_uri)
    set_github_output("deployment_state", deployment.provisioning_state)


if __name__ == "__main__":
    main()
