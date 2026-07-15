---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, adding a comment that says "CA review complete", and clicking the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff only — remove before sharing with students)*

| Check | Status | Notes |
|-------|--------|-------|
| Python Compatibility | 🟡 | While the tech stack predominantly uses Python (for unsupervised learning and data analysis), reliance on Azure for implementation may raise issues over Python compatibility. |
| Data Readiness | 🔴 | The project relies on multiple datasets, with one (StarScout) labeled, but estimates indicate a total size of over 10GB, posing immediate risks to data accessibility and usability. Students may need significant time to clean and manage data, resulting in an inefficient workflow. |
| Resource Check | 🟡 | Dataset readiness may be flagged as a critical risk regarding data ingestion (10+ GB). Loading a payload of this scale directly will trigger Out-of-Memory (OOM) fatal crashes and inefficient SQL queries against a 10 GB+ database, which risk rapidly depleting cloud credits. |

**Student Fit Score:** 5/10  
**Technical Depth Score:** 7/10  
**Overall Recommendation:** REVISE

**Advisor Feedback Draft:**
This project provides an excellent opportunity for Fellows to work on a project that is critical to MS business. However, to prevent immediate project failure due to technical debt and resource limitations, providng Fellows with a curated downsampled version of the 10 GB+ live data. This will ensures students can complete their exploratory data analysis and build their baseline model without getting stuck on infrastructure. Once their evaluation harness is stable, they maybe they can be introduced to a pre-computed database schema to scale up their graph networks and build a more scalable prototype.

---

# GitHub Repo Trust Signals

**Company / Org:** Microsoft  
**Challenge Advisor:** David Koleczek, dkoleczek@microsoft.com  
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About Microsoft

Microsoft is a global technology company that specializes in software development, hardware, and cloud services. Within the organization, we focus on leveraging AI and machine learning to enhance the user experience and security across our products.

---

## 🎯 The Challenge

### Project Summary
In this project, students will use real GitHub data and unsupervised learning techniques (anomaly detection, algorithms, and graph/network analysis) to build a repo-level trust score composed of multiple signals. Creating a trust metric will help Microsoft promote quality repos and find potentially malicious or manipulated repos faster.

### Success Criteria
Predictive power (target of 0.7 AUC on known repositories), user-friendly categorical labeling (likely trust/neutral/suspicious), and adherence to responsible AI principles regarding privacy and bias.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Understanding | Explore dataset, handle missing values, document findings |
| **October** | Model Development | Train baseline model, experiment with approaches, iterate |
| **November** | Evaluation & Presentation | Finalize model, prepare presentation, document results |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** StarScout dataset, gharchive.org (GitHub events), GitHub API  
**Format:** CSV  
**Size:** over 10gb  
**Location:** [Link to dataset or instructions for accessing it]

### Key Details
- Data includes the StarScout dataset (CSV files with labels), gharchive.org (raw GitHub events accessed via BigQuery), and live data from the GitHub API.
- Additional preprocessing may be needed to clean and format the datasets for analysis.
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification

**Recommended Libraries:**
- Unsupervised learning (anomaly detection, graph/network analysis)
- Heuristics
- GitHub API
- BigQuery
- StarScout dataset
- gharchive.org
- Google Colab
- Azure (optional).

**Evaluation Metrics:**
- AUC (Area Under the Curve), accuracy, precision/recall 

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [An Overview of GitHub as a Platform for Open Source Software](https://example.com)
- [The Importance of Trust Signals in Open Source](https://example.com)

**Technical Tutorials:**
- [Unsupervised Learning Techniques for Anomaly Detection](https://example.com)
- [Using GitHub API for Data Extraction](https://example.com)

**Code Examples:**
- [Link to a relevant GitHub repo](https://example.com)
- [Sample implementation on anomaly detection](https://example.com)

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Slack (Break Through Tech workspace)  
**Response time:** Within 48 hours on weekdays  

**Recommended Tools:**
- **Coding:** Google Colab
- **Collaboration:** GitHub, Notion
- **Virtual Meetings:** Zoom

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
