# Microsoft 3A: GitHub Repository Trust Signals
An interpretable, machine learning-based approach to evaluating the quality and trustworthiness of GitHub repositories using publicly available repository data.

> 💡 **Note for my team:** Hey everyone, it's Neha. Since we are quickly approaching our AI studio project deadline, I took the initiative to start our README. Below is my first draft. Please do not delete this note as it helps me remember that this README is unfinished.
---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Neha Modi        | @NehaM2025    | Trust signal evaluation, logistic regression modeling, ROC-AUC analysis, project documentation, and coordination.|

---

## 🎯 Project Highlights

- Engineered repository-level trust signals using publicly available GitHub metadata and activity.
- Evaluated individual signals using logistic regression and repeated stratified cross-validation.
- Applied data preprocessing techniques, including median imputation and feature standardization, within cross-validation folds.
- Used ROC-AUC to assess how effectively individual signals distinguish known-good repositories from uncertain repositories.
- Explored the limitations of automated repository assessment, including the risk of false classifications and bias against newer or less-established projects.

---

## 👩🏽‍💻 Setup and Installation

Ensure that [uv](https://docs.astral.sh/uv/getting-started/installation/) is installed and available. Check with:

```bash
uv --version
```

### Create the Virtual Environment and Install Dependencies

```bash
uv sync --frozen --all-extras --all-groups
```

To update existing dependencies:

```bash
uv sync -U --all-extras --all-groups
```

### Code Quality

Format, lint, and type check:

```bash
uv run ruff format && uv run ruff check --fix && uv run ty check
```

### Testing

Run tests:

```bash
uv run pytest
```

### GitHub API Token

Authentication is optional but recommended to increase GitHub API rate limits.

There are a few options:

- **Recommended:** Install the [GitHub CLI](https://cli.github.com/) and run `gh auth token`.
- Set a `GITHUB_TOKEN` environment variable.
- Create a [fine-grained personal access token](https://github.com/settings/personal-access-tokens/new):
  - Set the owner to your GitHub account.
  - Set repository access to "Public repositories."
  - No additional permissions are required.
  - Generate and configure the token as an environment variable.

**Important:** NEVER commit, include in a notebook, or share your token!

### Run the Analysis

[Insert verified commands for generating repository features, training models, and evaluating results.]

---

## 🏗️ Project Overview

The rapid adoption of AI coding agents has contributed to a significant increase in the number of repositories and development activity on GitHub. As the volume of available code grows, it becomes increasingly difficult to distinguish high-quality, actively maintained repositories from repositories that may have questionable or misleading trust signals.

Developed through the Break Through Tech AI Studio Fall 2026 program in collaboration with Microsoft, this project explores how publicly available GitHub data can be used to evaluate repository quality.

The objective is to develop a repository-level trust assessment using multiple signals derived from repository metadata, contributor activity, and development patterns. These signals are used to estimate the likelihood that a repository is high quality.

Importantly, the project is designed to supplement human judgment rather than replace it. A low or uncertain assessment should not automatically imply that a repository is malicious or unusable. The approach also considers how automated assessments might disadvantage newer or less-established repositories.

---

## 📊 Data Exploration

The project draws on publicly available GitHub data and repository datasets provided through the challenge.

### Repository Datasets

- **Known-good repositories (`data/good_repos.csv`):** Repositories identified as high-quality, well-used projects published by reputable organizations or users.
- **Uncertain repositories (`data/uncertain.csv`):** Repositories identified as potentially lower quality based on characteristics such as limited development history, engagement patterns, or marketing activity.

The initial repository lists were identified with the assistance of an AI agent and reviewed by humans.

### GitHub REST API

The GitHub REST API provides access to publicly available repository metadata and activity, including information about contributors, issues, pull requests, releases, and other development activity.

These data provide the foundation for extracting and evaluating repository-level trust signals.

### Feature Engineering

[Document the specific features implemented by the team, their definitions, and the data used to calculate them.]

### Potential Visualizations

- Repository dataset distributions.
- Feature distributions and comparisons between known-good and uncertain repositories.
- Correlations between repository characteristics.
- Individual trust signal performance comparisons.

---

## 🧠 Model Development

The project approaches repository trust assessment as a feature engineering and classification problem.

Rather than relying on a single repository characteristic, the approach examines multiple signals that may independently or collectively provide evidence of repository quality.

### Individual Signal Evaluation

To understand the predictive value of individual signals, we trained a separate logistic regression model for each signal.

The evaluation used:

- **Cross-validation:** Stratified 5-fold cross-validation.
- **Repeated evaluation:** 10 different random seeds to examine consistency across splits.
- **Preprocessing:** Median imputation and feature standardization performed within each training fold.
- **Evaluation metric:** ROC-AUC (Area Under the Receiver Operating Characteristic Curve).

This approach allows us to compare individual signals while examining how consistently they distinguish known-good repositories from uncertain repositories across different data splits.

### Additional Modeling Approaches

[Document any additional models, feature selection strategies, or combined scoring methodologies implemented by the team.]

---

## 📈 Results & Key Findings

[Insert final evaluation results, including individual signal performance, mean and standard deviation of ROC-AUC, and comparisons between signals.]

### Key Findings

- [Finding 1]
- [Finding 2]
- [Finding 3]

### Visualizations

[Insert cross-validation performance comparisons, signal evaluation plots, and other relevant visualizations.]

Potential visualizations include:

- Mean ROC-AUC comparison across individual trust signals.
- ROC-AUC variability across repeated cross-validation runs.
- Distribution of individual signal performance.

---

## 🚀 Next Steps

Repository trust cannot be established through a single metric. Individual signals may be misleading when evaluated without considering a repository's purpose, development stage, and context.

Potential limitations and areas for further development include:

- **New repository bias:** Newer or smaller repositories may have limited activity despite being legitimate, useful projects.
- **False classifications:** Unusual repository characteristics do not necessarily indicate malicious intent or poor quality.
- **Signal reliability:** Individual signals may have limited predictive power when considered independently.
- **Data limitations:** Public GitHub metadata does not capture every aspect of software quality, security, or maintainability.

Future development may include combining multiple signals into an overall repository-level trust score, evaluating the approach on additional repositories, and further examining the reliability and potential biases of individual signals.

---

## 📝 License

[Specify the license selected for the project.]

This project is licensed under the [License Name] License.

---

## 📄 References

- [GitHub REST API Documentation](https://docs.github.com/en/rest)
- [Six Million (Suspected) Fake Stars in GitHub](https://arxiv.org/)
- [StarScout Repository and Dataset](https://github.com/)

---

## 🙏 Acknowledgements

This project was completed as part of the Break Through Tech AI Studio Fall 2026 program in collaboration with Microsoft.

Special thanks to our Challenge Advisor, David Koleczek, and AI Studio Coach, Sai Duddu, for their guidance and support.


Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
