# GitHub Repo Trust Signals

**Company / Org:** Microsoft  
**Challenge Advisor:** David Koleczek
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🎯 The Challenge

### Project Summary

With the rapid adoption of coding agents, the number of repos and activity on GitHub has skyrocketed (in 2025 72M more repos were created than the year prior). One side effect of this is that it has become harder to distinguish high-quality or widely used repositories. In this project you will use publicly available GitHub data and unsupervised learning techniques (classification, anomaly detection, etc.) to build a repo-level score composed of multiple signals that indicates how confident we are that a repository is *high quality*. A signal can be thought of as a feature in machine learning. A simple example of a signal might be the ratio of stars compared to forks. A final repo-level score (ideally from 0 to 100, where 100 means maximum trust) could be the output of a machine learning model or a simpler algorithmic combination of the signals.

We provide a list of obviously good repos (think things like the Linux kernel, VSCode, etc.) as the positive class. We also provide a set of repos identified as *potentially* low-quality, malicious, host software that collects private data, and so on as the negative class, especially with respect to their amount of stars or known contributors, etc. The goal of the project is to then compute features from the repos and GitHub API and then create a machine learning model that predicts the liklihood of a repo being definitely *good*. The goal is *not* to definitively say that a repo is *bad* and should not be used.

### Success Criteria

1. There should be multiple (5+) high quality features derived from data available in a GitHub repo.
2. On the dataset of provided dataset of known-good and likely-bad repositories, each signal should have *some* predictive power (a reasonable target might be ~0.6 AUC). The overall model/score should have a higher AUC.
3. This solution needs to be done responsibly and preserve privacy. Even though all the data is public, analysis we can uncover things that a user never intended to make public. We also need to make sure are not introducing biases against new, but well-meaning users. For example, with the proliferation of repos, it has become harder to get attention, and our solution should not make that harder! Overall, the students should provide some writing on their analysis of this aspect and any limitations. We should target opting for an uncertain (neutral or unknown) label, rather than risking labeling a good repo as bad and vice versa.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

#### September - Data Understanding

- [ ] Literature review. See Resources to Get Started section below
- [ ] Understand the GitHub API. Exploring all the data you can get.
- [ ] Initial EDA on what signals might work.
- [ ] A plan for a classification mode and a candidate list of 5+ trust signals. 

#### October - Initial Model/Signal

- [ ] Dataset and evaluation split ready to go.
- [ ] Implement at least two signals
- [ ] Run the signal and get results for evaluation holdout set and try it on some live repos (pick some you find interesting!). 
- [ ] Analyze the results, produce a one-page report on your findings so far.

#### November - Evaluation and Presentation

- [ ] Working trust signals
- [ ] Be able to compute the score for a repo
- [ ] Final Presentation


> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

### Good and Uncertain Repo Lists

[data/](data/) contains two CSV files that each have one column of GitHub repositories in <owner>/<name> format. [data/good_repos.csv](.data/good_repos.csv) contains repositories that are known to be high-quality, well-used, and published by reputable organizations/users. [data/uncertain.csv](.data/uncertain.csv) contains repos that have a thin history, less engagement than is typical for the amount of stars and type of project, or contain significant marketing hype. Note that over time, some of these repos might get taken down or completely change. As we explore, we can consider adding to either dataset. These repos were discovered with the help of an AI agent and were all reviewed by humans.

### GitHub API

The [GitHub REST API](https://docs.github.com/en/rest) provides current public metadata about repositories, contributors, users, issues, pull requests, releases, and other GitHub activity. It will required for fetching information from live repos that we want to compute scores for. There is a sample client implemented at [src/gh_signals/github_client.py](src/gh_signals/github_client.py) and a notebook at [notebooks/001_github_client.ipynb](notebooks/001_github_client.ipynb) that shows an example of using it.


---

## 🛠️ Suggested Approach

**ML Problem Type:** Feature Engineering, Classification

**Recommended Libraries:**
- Unsupervised learning (anomaly detection, graph/network analysis)

**Evaluation Metrics:**
- AUC (Area Under the Curve), accuracy, precision/recall
- Visualizations of findings

---

## 📚 Resources to Get Started

### Background

The following resources will help your team understand the problem space and potential technical approaches for this project:

- [Six Million (Suspected) Fake Stars in GitHub](https://arxiv.org/abs/2412.13459)
- [Inside GitHub's Fake Star Economy](https://awesomeagents.ai/news/github-fake-stars-investigation/)
- [Vouch](https://github.com/mitchellh/vouch): A community trust management system based on explicit vouches to participate.

### StarScout Dataset

This is the original inspiration about the project. The [StarScout repository and dataset](https://github.com/hehao98/StarScout) contain results from a large-scale study of suspected fake GitHub stars from 2019 through 2024. StarScout identifies low-activity accounts and coordinated "lockstep" starring behavior, and includes repository-level results, monthly star counts, random comparison samples, and an external malware-campaign evaluation set. Its labels represent statistical suspicion rather than proof that an individual repository or account acted maliciously.

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Slack (Break Through Tech workspace)  
**Response time:** Within 48 hours on weekdays  

**Recommended Tools:**
- **Coding:** Local
- **Collaboration:** GitHub
- **Virtual Meetings:** Zoom

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)
