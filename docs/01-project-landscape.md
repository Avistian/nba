# 1. Project Landscape

## 1.1 The shift from heuristics to algorithms

Field sales — residential door-to-door (B2C) canvassing and territory-based commercial
(B2B) outside sales — has historically been driven by **heuristic decision-making**: static
maps, intuitive routing, and generalized neighborhood assumptions. Reps decided where to go
based on gut feel.

The physical constraints of outside sales make this expensive:

- Bounded by **geographic distance** and **travel time**.
- Limited **daylight / receptive hours** (residents are home in the evening, businesses
  during working hours).
- Human capital (the rep's time) is the single most expensive resource.

Two business facts motivate the move to algorithms:

- Organizations that rigorously manage their sales process generate up to **~28% more revenue**
  than peers.
- Structured automation typically yields **14–30% productivity gains**.

The response is a shift toward **algorithmic frameworks** that maximize return on physical
selling effort — with the **Next Best Action (NBA)** model at the center.

## 1.2 What "Next Best Action" means in the field

A **Next Best Action model** is a predictive engine that uses machine learning to recommend
the optimal *next* interaction with a specific prospect or account.

- In **digital / inside sales**, an NBA decision is simple: send an email vs. serve an ad.
- In **field sales**, NBA gains a **geospatial dimension**. It must decide:
  - Whether a rep should physically drive to a location at all.
  - The **optimal sequence** of physical visits.
  - How to **dynamically re-plan** based on real-time traffic, time windows, and prior
    interaction outcomes.

Building NBA for the field therefore fuses **predictive lead scoring**, **multi-channel
cadence automation**, and **operations research** (dynamic VRP / Traveling Salesperson
variants).

## 1.3 SPOTIO — the archetype platform

SPOTIO is the reference point for modern field-sales engagement software. It is deliberately
positioned as a **system of action** layered *on top of* a traditional CRM
(Salesforce, HubSpot, Zoho), which remains the **system of record**.

The "winning stack" pattern observed in the market:

- **Field execution platform** (e.g., SPOTIO) — mobile-first, geospatial workflows.
- **Enterprise CRM** — contacts, accounts, history.
- **Specialized optimization layers** — forecasting (e.g., Clari), conversation intelligence
  (e.g., Gong).

### Target verticals

Industries that depend on physical territory coverage:

- Telecommunications
- Solar energy
- Home improvement / roofing / storm restoration
- Home security
- Medical devices & pharmaceuticals
- Industrial distribution
- Financial services

Shared friction points: visualizing territories on a map, optimizing multi-stop routes,
GPS-verifying rep location, and stopping high-value leads from going cold between visits.

### Product tiers

| Tier | Selling motion | Key features | Audience |
|------|----------------|--------------|----------|
| **B2C** | High-volume residential canvassing | Rapid data capture, location-verified "pins," homeowner tracking, basic territory maps | Door-to-door solar, roofing, home security, storm restoration |
| **B2B** | Territory-based account management | Google Places B2B data (200+ business filters), multi-stop route optimization, account management | Telecom, industrial distribution, financial services, enterprise medical |
| **Custom / Enterprise** | Tailored multi-system deployments | Custom API work, non-standard territory mapping, custom metrics | Orgs with 25+ field reps |

## 1.4 AutoPlays — deterministic automation (the predecessor to NBA)

SPOTIO's **AutoPlays** are sequential, multi-channel cadences — a static, rules-based
predecessor to NBA. A representative or manager manually designs the sequence and enrolls a
contact. Example cadence:

- **Day 1** — initial call reminder + automated personalized email.
- **Day 2** — in-person visit reminder.
- **Day 3** — follow-up call.
- **Day 4** — automated text message.

Admins control intervals, restrict actions to business days, set time zones, and throttle
outbound email volume (to protect domain reputation). Effective sequences can be saved as
**templates** for the team.

### Why AutoPlays are *not* NBA

| AutoPlays (deterministic) | Next Best Action (probabilistic) |
|---------------------------|----------------------------------|
| Human designs the sequence | Model generates the sequence |
| Manual enrollment | Automatic enrollment based on state |
| Fixed intervals & channels | Adapts interval/channel from real-time feedback |
| No notion of expected value | Maximizes expected reward (value − cost) |
| No geography awareness | Integrates routing / proximity |

AutoPlays keep the human in the loop and remove admin overhead, **but lack the fluid,
probabilistic intelligence** of a real ML recommendation engine. They are a useful baseline
to build and then progressively replace (see [05-implementation-steps.md](05-implementation-steps.md)).

## 1.4a From passive tracker to active co-pilot

Today SPOTIO excels as a **system of record and field visibility**: it shows a map of pinned
territories, logs `Not Home` / `Pitched` / `Closed` statuses, and tracks rep GPS location.
But the **cognitive load of deciding *what to do next* still falls entirely on the rep** — who
stares at 500 pins and guesses where to start.

An NBA model converts this passive tracker into an **active, revenue-generating co-pilot**.
Instead of guessing, the rep receives a deterministic directive:

> *"Walk to **142 Elm Street** next. The probability of closing is highest because it is
> 5:00 PM, the household has lived there 10+ years, and their neighbor converted last week."*

The business effects:

- **Minimizes geographic dead-time** — fewer empty miles between doors.
- **Removes decision friction** — the rep sells instead of route-planning.
- **Lifts the floor, not just the ceiling** — average reps are guided toward the same choices
  top performers make intuitively, raising baseline team performance.

This is the difference between *showing data* and *prescribing the next action* — the precise
gap an NBA layer fills.

## 1.5 Pricing & value-capture economics

SPOTIO uses a high-touch enterprise SaaS model and **hides pricing** to force a sales demo.
Cost depends on headcount, integration complexity, and premium add-ons.

Indicative tiered pricing:

| Tier | Approx. price (per user / month) | Notes |
|------|----------------------------------|-------|
| Entry | ~$25 | Basic dashboards, territory assignment |
| Team | ~$39 | Minimum 5 users |
| Business | ~$69 | Route optimization, Google Places integration |
| Pro / Enterprise | ~$129+ | Custom quote |

Additional monetization via **add-on modules**: digital e-contracts, the *Lead Machine* data
engine, the *Multichannel Engagement* bundle, and the *DASH AI* co-pilot. Implementation fees
can reach ~$10,000, and a 10-person year-one total cost of ownership can approach ~$20,800
with add-ons.

Competitor contrast: **SalesRabbit** publishes transparent pricing (~$49–$75 per user/month),
positioning as a bundled, scalable alternative.

**Takeaway:** organizations tolerate premium fees because the ROI — higher conversion,
maximized territory yield, fuel savings — far outweighs the software cost. This is exactly the
value an in-house NBA layer aims to capture and amplify.

## 1.6 Strategic outlook

Digitizing analog workflows (static AutoPlays, basic map visibility) is now table stakes.
Durable advantage comes from building a **proprietary probabilistic intelligence layer** on
top of spatial CRM data — synthesizing OR (OR-Tools), ML reward modeling (LightGBM / SageMaker),
and geospatial infrastructure (PostGIS / AWS Location Service). The end state turns a rep from
a heuristic-driven wanderer into a **data-guided revenue engine**.
