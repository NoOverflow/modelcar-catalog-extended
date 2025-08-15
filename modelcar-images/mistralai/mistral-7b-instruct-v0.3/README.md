# Mistral-7b-instruct-v0.3

This model is untested and may have some issues.  Please feel free to contribute a PR to help fix any issues you encounter.

https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

quay.io/redhat-ai-services/modelcar-catalog:mistral-7b-instruct-v0.3

## Building Image

This model requires a user-token to authenticate with HuggingFace before pulling the model.  To generate a token, please refer to the [User access tokens](https://huggingface.co/docs/hub/en/security-tokens) documentation.

Once your token has been created, be sure to accept the terms and conditions for this model on the model home page.

```
podman build modelcar-images/mistral-7b-instruct-v0.3  \
    -t quay.io/redhat-ai-services/modelcar-catalog:mistral-7b-instruct-v0.3 \
    --build-arg HF_TOKEN="hf_..." \
    --platform linux/amd64
```

### Known Issues

#### Access to model is restricted

When attempting to download the model, you may get the following error message:

```
Cannot access gated repo for url https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db/config.json.
Access to model mistralai/Mistral-7B-Instruct-v0.3 is restricted and you are not in the authorized list. Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 to ask for access.
```

You must accept the terms and conditions on the model homepage.

#### No Space Left on Device

To build the model, you may require more storage to be allocated to the Podman VM.

It can help to remove all unnecessary images with the following command:

```
podman image prune --all
```
