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
With the rapid adoption of coding agents, the number of repos and activity on GitHub has skyrocketed (in 2025 72M more repos were created than the year prior), but this has also made it much easier to publish low-quality or malicious code. In this project you will use real GitHub data (gharchive.org which can also be accessed via BigQuery, the StarScout dataset, and the GitHub API) and unsupervised learning techniques (anomaly detection, algorithms, and graph/network analysis) to build a repo-level trust score composed of multiple signals. Creating a trust metric will help Microsoft promote quality repos and find potentially malicious or manipulated repos faster.

### Success Criteria

The team's solution should display the result in a user-friendly way: for example, a repo-level trust score where each signal likely is displayed such as: likely trust / neutral or unknown / likely suspicious.

On the dataset of provided dataset of known-good and likely-bad repositories, each signal should have predictive power (a reasonable target might be 0.7 AUC) perhaps measured only in the cases where we output trust or suspicious.

This solution needs to be done responsibly, preserve privacy (even though all the data is public, through analysis we can uncover things that the user never intended to make public), and make sure are not introducing biases against new, but well-meaning users. For example, with the proliferation of repos, it has become harder to get attention, and our solution should not make that harder! Overall, the students should provide some writing on their analysis of this aspect and any limitations. We should target opting for the uncertain (neutral or unknown) label, rather than risking labeling a good repo as bad and vice versa.

### Stretch Goals

Yes, absolutely, across a variety of directions. First, the dataset is essentially infinitely large. The team could spend more time exploring it especially for finding unsupervised signals using graph or network techniques (detecting trust of users based on their interactions with other repos, orgs, and users).

Second, the team could try to implement a more productionized implementation they can host. Azure for Students provides $100 in credits and free services.

Third, the team could try to publish their findings as a paper or blog.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | [Title] | The team should complete exploratory data analysis using the provided datasets and public APIs.<br>• Dataset inventory and schema understanding.<br>• Initial EDA on labeled suspicious repos and baseline repos.<br>• Literature review (Six Million (Suspected) Fake Stars on GitHub paper at https://arxiv.org/pdf/2412.13459) and a candidate list of 5+ trust signals. |
| October | [Title] | The team should have their first working signal and evaluation harness.<br>• Cleaned offline dataset and evaluation split ready to go.<br>• Implement at least one trust signal<br>• Run the signal and get results for evaluation holdout set and try it on some live repos.<br>• Analyze the initial results |
| November | [Title] | The team should have a working prototype with multiple trust signals that can be run against a live repo, which includes about 5 signals. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** See links below 
**Format:** CSV/ TSV,Database export (e.g., SQL dump)
**Size:** over 10gb  
**Location:** https://github.com/hehao98/StarScout/tree/main/data, https://www.gharchive.org/, https://docs.github.com/en/rest?apiVersion=2026-03-10

### Key Details
- StarScout has labels in csv files, gharchive is raw Github events, and the API can query live data

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification, Clustering, Unsupervised learning, Graph/network understanding, heuristics

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
