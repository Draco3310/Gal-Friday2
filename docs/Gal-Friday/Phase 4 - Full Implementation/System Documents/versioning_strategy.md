# Gal-Friday Documentation Versioning Strategy

## 1. Introduction

This document outlines the proposed versioning strategy for the Gal-Friday trading system's documentation. The primary purpose of this strategy is to ensure that all documentation artifacts remain synchronized with the evolving software, providing clarity, accuracy, and historical context for developers, maintainers, and potentially users of the system. A consistent versioning approach is crucial for managing changes, understanding the state of the system at different points in time, and facilitating effective collaboration.

## 2. Core Principles

The documentation versioning strategy is built upon the following core principles:

-   **Documentation as Code:** All documentation artifacts, including Markdown files (like this one), diagrams (e.g., Mermaid MMD files), and supporting assets, will be treated as integral parts of the Gal-Friday codebase. They will reside within the same Git repository as the application source code, under a dedicated `docs/` directory.
-   **Version Alignment:** Documentation versions will be closely aligned with software release versions. When a new version of Gal-Friday is released, the corresponding version of the documentation that reflects the features and state of that software release will also be tagged and made available.
-   **Branching Model Integration:** The documentation versioning and development process will integrate with the Git branching model used for the software development (e.g., GitFlow or a simplified version).
-   **Change Tracking:** All changes to documentation will be tracked through Git commits and Pull Requests, providing a clear history of updates, reviews, and approvals.

## 3. Proposed Versioning Scheme

-   **Semantic Versioning:** Both the Gal-Friday software and its accompanying documentation will adhere to Semantic Versioning (SemVer) practices. Versions will be specified in the format `MAJOR.MINOR.PATCH` (e.g., `v0.2.0`, `v1.0.0`, `v1.0.1`).
    -   `MAJOR` version incremented for incompatible API changes or significant architectural shifts.
    -   `MINOR` version incremented for adding functionality in a backward-compatible manner.
    -   `PATCH` version incremented for backward-compatible bug fixes or minor documentation corrections.
-   **Git Tags:** Specific versions of the documentation will be marked using Git tags.
    -   **Unified Tagging (Preferred):** A single Git tag (e.g., `v1.0.0`) will mark the commit that contains both the source code and the corresponding documentation for that release. This implies that documentation and code are released together from the same commit.
    -   **Separate Tagging (Alternative, if needed):** If, in specific circumstances, documentation needs to be updated independently for a released code version (e.g., urgent typo fixes not warranting a code patch), a separate tag like `docs-v1.0.1` could be considered, but the goal is to minimize divergence.

## 4. Git Branching Strategy for Documentation

The following Git branching strategy will be adopted for managing documentation development and releases, mirroring common software development practices:

-   **`main` (or `master`):**
    -   This branch contains the documentation corresponding to the **latest stable/production release** of Gal-Friday.
    -   It is the source of truth for users and developers looking for documentation related to the currently deployed version.
    -   Direct commits to `main` are generally discouraged; changes should come from merging `release` or `hotfix` branches.

-   **`develop`:**
    -   This branch contains the documentation for the **next planned release** of Gal-Friday, reflecting features and changes currently under development.
    -   Documentation for new features or significant updates is created on feature branches and merged into `develop` first.
    -   This branch represents the "bleeding edge" of documentation.

-   **`feature/<feature-name>-docs` (or `docs/<feature-name>`):**
    -   Short-lived branches created from `develop`.
    -   Used for writing documentation for new software features, making substantial revisions to existing documentation, or adding new documentation sections (e.g., a new module document).
    -   These branches are merged back into `develop` via Pull Requests, which should include a review of the documentation changes.

-   **`release/<version>-docs` (Optional, but Recommended):**
    -   Branched from `develop` when preparing for a new software release (e.g., `release/v1.1.0-docs`).
    -   Used for final documentation reviews, proofreading, copy-editing, fixing minor documentation issues specific to the release, and ensuring all content is up-to-date and accurate for the version being released.
    -   Version numbers within the documentation (if any are explicitly mentioned in text) are updated here.
    -   Once finalized, this branch is merged into `main` (and tagged) and also back into `develop` (to ensure `develop` also gets these final fixes).

-   **`hotfix/<fix-name>-docs` (or `fix/docs/<issue-id>`):**
    -   Created from `main` to address urgent errors, typos, or critical omissions in the documentation of a production release.
    -   These changes are minimal and focused solely on correcting the documentation for the existing release.
    -   Once the fix is complete, the hotfix branch is merged back into `main` (and a new patch tag, e.g., `v1.0.1`, might be created) and also merged into `develop` to ensure the fix is carried forward.

## 5. Change Tracking and History

Maintaining a clear history of documentation changes is crucial:

-   **Commit Messages:** Git commit messages for documentation changes must be clear, descriptive, and follow any project-wide commit message conventions. They should ideally reference the specific part of the documentation being changed or the related software feature/issue.
    -   Example: `docs: Update RiskManager module DFD interactions (Issue #123)`
-   **Pull Requests (PRs):** All significant documentation changes (especially those on `feature/*-docs` or `hotfix/*-docs` branches) should be submitted and merged via Pull Requests.
    -   PRs provide a venue for review and discussion of documentation changes, similar to code reviews.
    -   PR descriptions should summarize the changes made to the documentation.
-   **Changelog (`CHANGELOG.md`):**
    -   A `CHANGELOG.md` file will be maintained within the `docs/Gal-Friday/` directory (or at the root of the `docs/` folder).
    -   For each software version release, a corresponding section will be added to the changelog summarizing significant updates, additions, or corrections made to the documentation.
    -   Where appropriate, entries can link to the specific documentation files or sections that were updated.
    -   This provides a human-readable summary of documentation evolution alongside software evolution.

## 6. Integration with Documentation Publishing (Future Considerations)

While the initial focus is on maintaining Markdown files within the Git repository, future plans may involve publishing this documentation as a static website (e.g., using tools like MkDocs, Sphinx with MyST Parser, Docsify, ReadtheDocs, or GitBook). When this is implemented, the versioning strategy will extend to the publishing pipeline:

-   **Version-Specific Builds:** The publishing pipeline (e.g., GitHub Actions, GitLab CI) will be configured to build and deploy the documentation from specific Git tags (e.g., `v1.0.0`, `v1.1.0`) corresponding to releases. This ensures that the published documentation accurately reflects the state of the system for that version.
-   **Version Display:** The published documentation website must clearly display the version of the documentation being viewed (e.g., in the site header or sidebar).
-   **Version Switching:** For users needing to access documentation for older or different versions of Gal-Friday, the published site should ideally provide a mechanism (e.g., a version dropdown menu) to switch between different released versions of the documentation. This often involves publishing each version to a distinct URL path (e.g., `/docs/v1.0/`, `/docs/v1.1/`).
-   **"Latest" Version:** A URL pointing to the documentation for the `develop` branch (or latest pre-release) might also be provided for those interested in upcoming features, clearly marked as "development" or "unreleased."

## 7. Review and Updates to this Strategy

This documentation versioning strategy is a living document. It will be reviewed periodically and updated as the Gal-Friday project's needs, team size, or development practices evolve. Any proposed changes to this strategy should themselves be discussed and version-controlled.
