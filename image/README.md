# Production GHCR release pipeline

This bundle targets `athithya-sakthivel/secure-llm-deploy`, publishes to
`ghcr.io/athithya-sakthivel/secure-llm-deploy`, and has no GitHub Environments.
The workflow runs only for pushes to `main` and version-like tags matching `v*`.
All third-party workflow actions are pinned to full commit SHAs with the
corresponding release version recorded in a comment.

## 1. Create the two repository secrets

Create a **classic** GitHub Personal Access Token with only `write:packages`
for publishing this image. Add `read:packages` only if your use case also needs
to pull private packages. Do not grant the `repo` scope merely because the UI
suggests it. Follow your organization/account's SSO requirements if applicable.

From the repository root, use GitHub CLI. The token is read without echo and is
not embedded in the command line or shell history:

```bash
read -rsp 'GHCR_PAT: ' GHCR_PAT
printf '\n'
printf '%s' "$GHCR_PAT" | gh secret set GHCR_PAT
unset GHCR_PAT
gh secret set GHCR_USER_NAME --body 'athithya-sakthivel'
````

Check that the repository is `athithya-sakthivel/secure-llm-deploy` before running the commands. Do not create GitHub Environments.

GitHub recommends `GITHUB_TOKEN` for GHCR packages linked to the same repository when possible. This workflow uses the two requested secrets to authenticate to GHCR. The workflow itself also needs `packages: write` for artifact attestations. If the image package already exists, open the package settings and confirm that it is connected to this repository and that Actions has package write access. The `org.opencontainers.image.source` label is also included to help link the image package to the source repository.

## 2. Install the files

Copy `image/` into the repository's existing `image/` directory and copy the `.github/` contents into the repository's `.github/` directory. Review and commit all changes, including `image/model.lock`. Do not commit tokens.

The default branch must be named `main`; otherwise change the branch filter in `.github/workflows/release.yml`. The workflow only publishes when run in the specified repository. The `IMAGE_NAME` value is intentionally explicit rather than derived from a potentially mixed-case repository name.

## 3. What CI does

1. Checks that the required secrets are present and the configured user is the expected lowercase GHCR namespace.
2. Builds `linux/amd64` with the digest-pinned vLLM base, the committed model lock, BuildKit provenance, BuildKit SBOM, and GitHub Actions build cache.
3. Pushes a source-SHA tag to GHCR and captures the pushed registry digest.
4. Scans that precise `image@sha256:...` reference with Trivy. HIGH or CRITICAL OS/library vulnerabilities fail the job, including findings without a fix.
5. Generates an SPDX JSON SBOM for the same digest.
6. Signs the image digest with keyless Cosign and verifies that the signing certificate was issued to this repository's workflow on `main` or a `v*` tag.
7. On public repositories, publishes build-provenance and SBOM attestations to GHCR. The SPDX SBOM and Trivy SARIF report are retained as workflow artifacts. Public repositories also publish the Trivy SARIF results to GitHub Code Scanning.
8. Prints the exact deployment reference to the Actions run summary.

The image tag is a convenience, not an integrity boundary. The digest is the canonical identity. Deploy the exact `ghcr.io/athithya-sakthivel/secure-llm-deploy@sha256:...` reference emitted by the workflow, and require signature/attestation verification in your deployment process.

## 4. Verify after a release

Set `IMAGE` to the exact digest printed in the workflow summary:

```
export IMAGE='ghcr.io/athithya-sakthivel/secure-llm-deploy@sha256:REPLACE_WITH_DIGEST'

cosign verify \
  --certificate-identity-regexp '^https://github\.com/athithya-sakthivel/secure-llm-deploy/\.github/workflows/release\.yml@refs/(heads/main|tags/v.*)$' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  "$IMAGE"

# GitHub-hosted artifact attestation verification (also checks the source repo):
gh attestation verify "oci://${IMAGE}" --repo athithya-sakthivel/secure-llm-deploy

# Verify the SPDX SBOM attestation
gh attestation verify "oci://${IMAGE}" \
  --repo athithya-sakthivel/secure-llm-deploy \
  --predicate-type https://spdx.dev/Document/v2.3
```

GitHub artifact-attestation steps are deliberately skipped for private repositories to avoid failing on plans where GitHub does not offer artifact attestations for private/internal repositories. GitHub's current requirements make those attestations available for private/internal repos only on GitHub Enterprise Cloud. The GitHub Code Scanning SARIF upload is likewise limited to public repositories in this workflow; the report remains available as a workflow artifact for private repositories. BuildKit provenance and SBOM generation, plus Cosign image signing, still run for private repositories. To use GitHub artifact attestations for a private repository, first confirm your plan entitlement and then remove the `if: github.event.repository.private == false` conditions from the two `actions/attest` steps.

## 5. Operational requirements outside CI

- Pin the deployed image by digest and verify its signature before deployment.
- Keep the image package private unless public distribution is intended.
- Restrict production network ingress; loopback-only host binding is suitable for a local smoke test, not a remote API endpoint. Use TLS and an auth/rate-limit gateway for remote consumers.
- Supply `VLLM_API_KEY` or enforce authentication at a trusted gateway. Avoid exposing the raw vLLM port directly to the internet.
- Do not set `VLLM_TRUST_REMOTE_CODE=1` except for a reviewed model that requires it. Treat model upgrades as supply-chain changes and review `model.lock` diffs.
- Test on the exact production GPU/driver stack, including startup time, memory pressure, throughput, context length, and graceful shutdown.
- Monitor for newly disclosed vulnerabilities and update pinned base images and action SHAs by reviewed Dependabot PRs. A SHA pin makes an action immutable; it does not automatically keep it patched.
- The scanner gate is intentionally strict. If it fails, review findings and update/replace the affected component or document a narrow, expiring exception with a reason rather than globally ignoring all unfixed issues.

## Upstream documentation checked

- GitHub GHCR authentication and package access: [https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- GitHub container publishing action guidance: [https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)
- GitHub artifact attestations: [https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations)
- Docker image build best practices and digest pinning: [https://docs.docker.com/build/building/best-practices/](https://docs.docker.com/build/building/best-practices/)
- Cosign container signing: [https://docs.sigstore.dev/cosign/signing/signing_with_containers/](https://docs.sigstore.dev/cosign/signing/signing_with_containers/)
- Trivy GitHub Action: [https://github.com/aquasecurity/trivy-action](https://github.com/aquasecurity/trivy-action)
- Build Push Action: [https://github.com/docker/build-push-action](https://github.com/docker/build-push-action)

```

One implementation caveat: these are the complete files from the generated bundle, not a claim that a real GHCR release has succeeded. Before relying on this in production, run the workflow in the target repository and confirm the pinned action versions, attestation inputs, token permissions, and registry behavior against the current upstream documentation. The actual GPU startup and inference path also needs testing on your deployment hardware.
```


