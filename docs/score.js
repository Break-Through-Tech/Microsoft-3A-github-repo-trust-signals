/* Browser scorer. The formulas match gh_signals.score for the saved logistic model. */
const TrustScore = (() => {
  const MIN_AGE_DAYS = 90;
  const MIN_STARS = 50;
  const STAR_COUNT_FLOOR = 1000;
  const ESTABLISHED_AT = 70;
  const MIXED_AT = 40;
  const STAR_HISTORY_TIMEOUT_MS = 4000;
  const CONTRIBUTION_EPSILON = 0.05;
  const SPECS = [
    ["stargazers_count", "Stars", "count"],
    ["forks_count", "Forks", "count"],
    ["subscribers_count", "Watchers", "count"],
    ["open_issues_count", "Open issues", "count"],
    ["size", "Repository size", "size"],
    ["age_days", "Age in days", "days"],
    ["days_since_push", "Days since last push", "days"],
    ["fork_star_ratio", "Forks per star", "ratio"],
    ["subscriber_star_ratio", "Watchers per star", "ratio"],
    ["issues_per_star", "Open issues per star", "ratio"],
    ["stars_per_day", "Stars per day", "ratio"],
    ["has_license", "License detected", "flag"],
    ["is_org_owned", "Organization owner", "flag"],
    ["star_to_follower_ratio", "Stars per owner follower", "ratio"],
    ["peak_stars", "Stars on the busiest day", "count"],
    ["max_day_share", "Busiest day's share of stars", "share"],
    ["active_day_ratio", "Days with at least one star", "share"],
    ["burstiness", "Star burstiness", "burst"],
  ];
  const STAR_FEATURES = new Set(["peak_stars", "max_day_share", "active_day_ratio", "burstiness"]);
  const DAY_MS = 86400000;

  function parseRepoUrl(text) {
    const raw = text.trim();
    if (!raw) {
      throw new Error("Enter a GitHub repository URL.");
    }
    const segment = /^[A-Za-z0-9_.-]+$/;
    const join = (owner, name) => {
      let repoName = name.endsWith(".git") ? name.slice(0, -4) : name;
      if (owner === "." || owner === ".." || repoName === "." || repoName === ".." || !segment.test(owner) || !segment.test(repoName)) {
        throw new Error("That repository name contains characters GitHub does not allow.");
      }
      return `${owner}/${repoName}`;
    };
    if (!raw.includes("://") && !raw.toLowerCase().startsWith("github.com") && !raw.toLowerCase().startsWith("www.github.com")) {
      const parts = raw.split("/").filter(Boolean);
      if (parts.length !== 2) {
        throw new Error("Use a github.com URL or an owner/name pair, such as microsoft/vscode.");
      }
      return join(parts[0], parts[1]);
    }
    const withScheme = raw.includes("://") ? raw : `https://${raw}`;
    let parsed;
    try {
      parsed = new URL(withScheme);
    } catch {
      throw new Error("Use a public github.com repository URL.");
    }
    let host = parsed.hostname.toLowerCase();
    if (host.startsWith("www.")) {
      host = host.slice(4);
    }
    if (host !== "github.com") {
      throw new Error("Use a public github.com repository URL.");
    }
    const parts = parsed.pathname.split("/").filter(Boolean);
    if (parts.length < 2) {
      throw new Error("That URL does not include an owner and a repository name.");
    }
    return join(parts[0], parts[1]);
  }

  function pythonRound(value) {
    const sign = value < 0 ? -1 : 1;
    const absolute = Math.abs(value);
    const floor = Math.floor(absolute);
    const fraction = absolute - floor;
    if (Math.abs(fraction - 0.5) < 1e-10) {
      const even = floor % 2 === 0 ? floor : floor + 1;
      return sign * even;
    }
    return sign * Math.round(absolute);
  }

  function sigmoid(logit) {
    if (logit >= 30) {
      return 1;
    }
    if (logit <= -30) {
      return 0;
    }
    return 1 / (1 + Math.exp(-logit));
  }

  function predict(model, features) {
    const contributions = [];
    let logit = model.intercept;
    model.feature_names.forEach((name, index) => {
      const raw = features[name];
      const filled = raw === null || raw === undefined ? model.medians[index] : raw;
      const scale = model.scales[index] || 1;
      const contribution = model.coefficients[index] * ((filled - model.means[index]) / scale);
      contributions.push(contribution);
      logit += contribution;
    });
    return { probability: sigmoid(logit), contributions };
  }

  function applyStarFloor(features, contributions, featureNames) {
    const index = featureNames.indexOf("stargazers_count");
    if (index < 0) {
      return { contributions: contributions.slice(), floored: false };
    }
    const stars = features.stargazers_count;
    const adjusted = contributions.slice();
    if (stars === null || stars === undefined || stars < STAR_COUNT_FLOOR || adjusted[index] >= 0) {
      return { contributions: adjusted, floored: false };
    }
    adjusted[index] = 0;
    return { contributions: adjusted, floored: true };
  }

  function formatValue(kind, value, available) {
    if (!available || value === null || value === undefined) {
      return "Not loaded";
    }
    if (kind === "flag") {
      return value >= 0.5 ? "Yes" : "No";
    }
    if (kind === "share") {
      return `${(value * 100).toFixed(1)}%`;
    }
    if (kind === "burst") {
      return value.toFixed(2);
    }
    if (kind === "ratio") {
      return Math.abs(value) >= 100 ? value.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) : value.toFixed(3);
    }
    const rendered = pythonRound(value).toLocaleString("en-US");
    return kind === "size" ? `${rendered} KB` : rendered;
  }

  function decideTier(points, ageDays, stars) {
    const tooNew = ageDays < MIN_AGE_DAYS;
    const tooSmall = stars < MIN_STARS;
    if (tooNew && tooSmall) {
      return [null, "limited_evidence", "Limited evidence", "This repository is under 90 days old and has fewer than 50 stars. The numeric score is withheld so age and audience size are not read as a lack of trust."];
    }
    if (tooNew) {
      return [null, "limited_evidence", "Limited evidence", "This repository is under 90 days old. The numeric score is withheld so age is not read as a lack of trust."];
    }
    if (tooSmall) {
      return [null, "limited_evidence", "Limited evidence", "This repository has fewer than 50 stars. The numeric score is withheld so a smaller audience is not read as a lack of trust."];
    }
    if (points >= ESTABLISHED_AT) {
      return [points, "established", "Established", "Public metadata resembles the known-good repositories used to train this score."];
    }
    if (points >= MIXED_AT) {
      return [points, "mixed", "Mixed", "Public metadata is mixed. Some signals resemble known-good repositories and some do not."];
    }
    return [points, "limited_evidence", "Limited evidence", "Public metadata does not resemble the known-good repositories. This is limited evidence, not a verdict that the repository is harmful."];
  }

  function buildScore(model, fullName, features, warnings) {
    let { probability, contributions } = predict(model, features);
    const floored = applyStarFloor(features, contributions, model.feature_names);
    contributions = floored.contributions;
    if (floored.floored) {
      probability = sigmoid(model.intercept + contributions.reduce((sum, value) => sum + value, 0));
    }
    const points = Math.min(100, Math.max(0, pythonRound(probability * 100)));
    const ageDays = Math.trunc(features.age_days || 0);
    const stars = Math.trunc(features.stargazers_count || 0);
    const [score, tier, tierLabel, summary] = decideTier(points, ageDays, stars);
    const shownWarnings = warnings.slice();
    if (floored.floored) {
      shownWarnings.push("This repository has at least 1,000 stars, so the star count does not lower the score.");
    }
    const signals = model.feature_names.map((name, index) => {
      const spec = SPECS.find((item) => item[0] === name);
      const value = features[name];
      const available = value !== null && value !== undefined;
      const contribution = contributions[index];
      let direction = "neutral";
      let note = "Close to the typical training value, so it barely moves the score.";
      if (!available) {
        direction = "unavailable";
        note = "Not loaded. The score uses the typical value from the training set.";
      } else if (contribution > CONTRIBUTION_EPSILON) {
        direction = "up";
        note = "Moves the score up compared with the repositories used to train this model.";
      } else if (contribution < -CONTRIBUTION_EPSILON) {
        direction = "down";
        note = "Moves the score down compared with the repositories used to train this model.";
      }
      if (floored.floored && name === "stargazers_count" && available) {
        note = "At least 1,000 stars. The training set is mostly more famous projects, so this count is not allowed to lower the score.";
      }
      return {
        name,
        label: spec[1],
        value,
        display_value: formatValue(spec[2], value, available),
        direction,
        note,
        available,
      };
    });
    const snapshotDay = model.snapshot.slice(0, 10);
    return {
      full_name: fullName,
      repo_url: `https://github.com/${fullName}`,
      score,
      tier,
      tier_label: tierLabel,
      summary,
      signals,
      warnings: shownWarnings,
      disclaimer: `This score summarizes public GitHub metadata from the training snapshot on ${snapshotDay}. A withheld or low score means limited evidence, not that the repository is harmful or malicious. Repository contents are not downloaded or run.`,
      trained_through: model.snapshot,
    };
  }

  function utcMidnight(ms) {
    const date = new Date(ms);
    return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
  }

  function dailyCounts(weeks, createdMs, endMs) {
    const counts = new Map();
    const start = utcMidnight(createdMs);
    const end = utcMidnight(endMs);
    for (let day = start; day <= end; day += DAY_MS) {
      counts.set(day, 0);
    }
    weeks.forEach((week) => {
      const weekStart = utcMidnight(week.week * 1000);
      week.days.forEach((stars, offset) => {
        const day = weekStart + offset * DAY_MS;
        if (counts.has(day)) {
          counts.set(day, counts.get(day) + stars);
        }
      });
    });
    return counts;
  }

  function starTiming(repo, weeks, asOfMs) {
    if (!weeks || weeks.length === 0) {
      return null;
    }
    const yesterday = utcMidnight(asOfMs) - DAY_MS;
    const created = utcMidnight(Date.parse(repo.created_at));
    if (yesterday < created) {
      return null;
    }
    const counts = dailyCounts(weeks, Date.parse(repo.created_at), yesterday);
    const values = Array.from(counts.values());
    const total = values.reduce((sum, value) => sum + value, 0);
    if (total === 0) {
      return null;
    }
    let peak = 0;
    values.forEach((value) => {
      if (value > peak) {
        peak = value;
      }
    });
    const mean = total / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    const std = Math.sqrt(variance);
    const active = values.filter((value) => value > 0).length / values.length;
    return {
      peak_stars: peak,
      max_day_share: peak / total,
      active_day_ratio: active,
      burstiness: (std - mean) / (std + mean),
    };
  }

  function liveFeatures(repo, owner, weeks, asOfMs) {
    const stars = repo.stargazers_count;
    const ageDays = Math.max(0, Math.floor((asOfMs - Date.parse(repo.created_at)) / DAY_MS));
    const timing = starTiming(repo, weeks, asOfMs);
    const ratio = (numerator, denominator) => (denominator ? numerator / denominator : null);
    return {
      stargazers_count: Number(stars),
      forks_count: Number(repo.forks_count),
      subscribers_count: Number(repo.subscribers_count),
      open_issues_count: Number(repo.open_issues_count),
      size: Number(repo.size),
      age_days: ageDays,
      days_since_push: Math.max(0, Math.floor((asOfMs - Date.parse(repo.pushed_at)) / DAY_MS)),
      fork_star_ratio: ratio(repo.forks_count, stars),
      subscriber_star_ratio: stars ? repo.subscribers_count / stars : null,
      issues_per_star: ratio(repo.open_issues_count, stars),
      stars_per_day: stars / Math.max(ageDays, 1),
      has_license: repo.license ? 1 : 0,
      is_org_owned: repo.owner && repo.owner.type === "Organization" ? 1 : 0,
      star_to_follower_ratio: owner && owner.followers ? stars / owner.followers : null,
      peak_stars: timing ? timing.peak_stars : null,
      max_day_share: timing ? timing.max_day_share : null,
      active_day_ratio: timing ? timing.active_day_ratio : null,
      burstiness: timing ? timing.burstiness : null,
    };
  }

  function describeUsage(elapsedMs, requests, remaining, limit) {
    const took = elapsedMs < 100 ? "Took under 0.1 seconds." : elapsedMs < 10000 ? `Took ${(elapsedMs / 1000).toFixed(1)} seconds.` : `Took ${Math.round(elapsedMs / 1000)} seconds.`;
    const word = requests === 1 ? "request" : "requests";
    let sentence = `${took} Used ${requests} GitHub ${word}.`;
    if (remaining !== null && limit !== null) {
      sentence += ` ${Number(remaining).toLocaleString("en-US")} of ${Number(limit).toLocaleString("en-US")} remain this hour.`;
    }
    return sentence;
  }

  function failureMessage(status) {
    if (status === 404) {
      return "GitHub could not find that repository. It may be private, renamed, or missing.";
    }
    if (status === 403 || status === 429) {
      return "GitHub rate limit reached. Try again in a little while.";
    }
    return "GitHub returned an error. Try again in a moment.";
  }

  async function githubGet(path, tally, signal) {
    const response = await fetch(`https://api.github.com/${path}`, {
      headers: {
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      signal,
    });
    tally.requests += 1;
    const remaining = response.headers.get("x-ratelimit-remaining");
    const limit = response.headers.get("x-ratelimit-limit");
    if (remaining !== null) {
      tally.remaining = remaining;
    }
    if (limit !== null) {
      tally.limit = limit;
    }
    if (!response.ok) {
      const error = new Error(failureMessage(response.status));
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  async function loadStarWeeks(fullName, tally) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), STAR_HISTORY_TIMEOUT_MS);
    const weeks = [];
    try {
      for (let page = 1; page <= 100; page += 1) {
        const batch = await githubGet(
          `repos/${fullName}/stargazers/history?per_page=30&page=${page}`,
          tally,
          controller.signal,
        );
        if (!Array.isArray(batch) || batch.length === 0) {
          break;
        }
        weeks.push(...batch);
      }
      return { weeks, warning: null };
    } catch (error) {
      if (error.name === "AbortError") {
        return { weeks: null, warning: "Star timing did not return in time, so those signals use a typical training value." };
      }
      return { weeks: null, warning: "Star timing is not available for this repository, so those signals use a typical training value." };
    } finally {
      clearTimeout(timer);
    }
  }

  async function loadModel() {
    const response = await fetch("trust_model.json");
    if (!response.ok) {
      throw new Error("The trained model file is missing.");
    }
    return response.json();
  }

  async function scoreRepository(text) {
    const requested = parseRepoUrl(text);
    const started = performance.now();
    const model = await loadModel();
    const tally = { requests: 0, remaining: null, limit: null };
    const repo = await githubGet(`repos/${requested}`, tally);
    const warnings = [];
    if (repo.archived) {
      warnings.push("This repository is archived.");
    }
    if (repo.fork) {
      warnings.push("This repository is a fork.");
    }
    let owner = null;
    try {
      owner = await githubGet(`users/${repo.owner.login}`, tally);
    } catch {
      warnings.push("The owner profile could not be loaded, so stars per owner follower uses a typical training value.");
    }
    const history = await loadStarWeeks(repo.full_name, tally);
    if (history.warning) {
      warnings.push(history.warning);
    }
    const asOf = Date.now();
    const features = liveFeatures(repo, owner, history.weeks, asOf);
    if (!history.warning && STAR_FEATURES.size && [...STAR_FEATURES].every((name) => features[name] === null)) {
      warnings.push("Star timing is not available for this repository, so those signals use a typical training value.");
    }
    const result = buildScore(model, repo.full_name, features, warnings);
    const elapsed = Math.max(0, Math.round(performance.now() - started));
    result.elapsed_ms = elapsed;
    result.github_requests = tally.requests;
    result.usage_summary = describeUsage(elapsed, tally.requests, tally.remaining, tally.limit);
    return result;
  }

  return { parseRepoUrl, buildScore, liveFeatures, scoreRepository, pythonRound };
})();

if (typeof module !== "undefined") {
  module.exports = TrustScore;
}
