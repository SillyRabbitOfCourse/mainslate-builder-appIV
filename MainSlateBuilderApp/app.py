# ================================================================
#                    IMPORTS & GLOBAL CONFIG
# ================================================================

import random
from typing import List, Dict, Tuple
import pandas as pd
import streamlit as st
import re

NUM_LINEUPS = 40
SALARY_CAP = 50000
MIN_SALARY = 49000
RANDOM_SEED = 42

SLOT_ORDER = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
FLEX_ELIGIBLE = {"RB", "WR", "TE"}

STACK_TEAMS: List[str] = []
STACK_EXPOSURES: Dict[str, float] = {}
STACK_REQUIRED: Dict[str, List[str]] = {}
STACK_OPTIONAL: Dict[str, Dict[str, float]] = {}
STACK_MIN_MAX: Dict[str, Tuple[int, int]] = {}

STACK_RUNBACK_TEAMS: Dict[str, str] = {}
STACK_RUNBACKS: Dict[str, Dict[str, float]] = {}
STACK_RUNBACK_MIN_MAX: Dict[str, Tuple[int, int]] = {}

STACK_INCLUDE_DST: Dict[str, bool] = {}
STACK_DST_PERCENT: Dict[str, float] = {}

MINI_STACKS: List[Dict] = []

TEAM_FILTER_MODE: Dict[str, str] = {}
TEAM_FILTER_KEEP: Dict[str, List[str]] = {}
TEAM_FILTER_EXCLUDE: Dict[str, List[str]] = {}

MAX_ATTEMPTS_PER_LINEUP = 20000

# ================================================================
#                          DATA LOADING
# ================================================================

def load_player_pool(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    df["Salary"] = df["Salary"].astype(int)
    return df

# ================================================================
#                     OPPONENT EXTRACTION
# ================================================================

def extract_opponents(df: pd.DataFrame) -> Dict[str, str]:
    matchup_map = {}
    for _, row in df.iterrows():
        match = re.match(r"([A-Z]+)@([A-Z]+)", row["Game Info"])
        if match:
            away, home = match.group(1), match.group(2)
            matchup_map[away] = home
            matchup_map[home] = away
    return matchup_map

# ================================================================
#                    GLOBAL PLAYER POOL FILTERING
# ================================================================

def apply_global_team_filters(df, modes, keep_map, exclude_map):
    out = df.copy()
    for team, mode in modes.items():
        team_rows = out["TeamAbbrev"] == team
        if mode == "remove_team":
            out = out[~team_rows]
        elif mode == "keep_only":
            keep = keep_map.get(team, [])
            out = out[~team_rows | out["Name"].isin(keep)]
        elif mode == "exclude_only":
            exc = exclude_map.get(team, [])
            out = out[~team_rows | ~out["Name"].isin(exc)]
    return out.reset_index(drop=True)





# ================================================================
#                     STACK OPTIONAL SPRINKLES
# ================================================================

def sample_optional_players(team: str):
    chosen = []
    for p, pct in STACK_OPTIONAL.get(team, {}).items():
        if random.random() < pct:
            chosen.append(p)
    return chosen

# ================================================================
#                     RUNBACK POOL
# ================================================================

def get_runback_pool(df, opp):
    return df[
        (df["TeamAbbrev"] == opp) &
        (df["Position"] != "QB")
    ].reset_index(drop=True)

# ================================================================
#                     POSITION SPLIT
# ================================================================

def position_split(df):
    groups = {}
    for pos, grp in df.groupby("Position"):
        groups[pos] = grp.reset_index(drop=True)
    return groups

# ================================================================
#              MINI-STACK STATE FOR EXPOSURE
# ================================================================

def init_mini_stack_state(num_lineups, mini_stacks):
    rules = []
    for rule in mini_stacks:
        r = dict(rule)
        r["remaining"] = int(num_lineups * rule["exposure"])
        rules.append(r)
    return rules

def mini_rule_applicable(rule, primary, stack_teams, runbacks):
    s = set(stack_teams)
    r = set(runbacks.values())
    if rule["type"] == "same_team":
        return (
            rule["team"] != primary and
            rule["team"] not in s and
            rule["team"] not in r
        )
    t1, t2 = rule["team1"], rule["team2"]
    return (
        t1 != primary and t2 != primary and
        t1 not in s and t2 not in s and
        t1 not in r and t2 not in r
    )




# ================================================================
#                MINI STACK PLAYER PICKER
# ================================================================

def pick_mini_stack_players(rule, df, used_ids):
    if rule["type"] == "same_team":
        t = rule["team"]
        p1, p2 = rule["players"]
        r1 = df[(df["Name"] == p1) & (df["TeamAbbrev"] == t)]
        r2 = df[(df["Name"] == p2) & (df["TeamAbbrev"] == t)]
        if r1.empty or r2.empty:
            return None
        P1, P2 = r1.iloc[0], r2.iloc[0]
        if P1.ID in used_ids or P2.ID in used_ids:
            return None
        return [P1, P2]

    # Opposing teams
    t1, t2 = rule["team1"], rule["team2"]
    p1, p2 = rule["players"]
    r1 = df[(df["Name"] == p1) & (df["TeamAbbrev"] == t1)]
    r2 = df[(df["Name"] == p2) & (df["TeamAbbrev"] == t2)]
    if r1.empty or r2.empty:
        return None
    P1, P2 = r1.iloc[0], r2.iloc[0]
    if P1.ID in used_ids or P2.ID in used_ids:
        return None
    return [P1, P2]

# ================================================================
#                MODE B WEIGHTED RUNBACK SELECTION
# ================================================================

def select_runbacks_for_stack(team, df):
    opp = STACK_RUNBACK_TEAMS.get(team, "")
    min_req, max_req = STACK_RUNBACK_MIN_MAX.get(team, (0, 0))

    if not opp:
        return [] if min_req == 0 else None

    pool = get_runback_pool(df, opp)
    if pool.empty:
        return [] if min_req == 0 else None

    weights = STACK_RUNBACKS.get(team, {})
    pool = pool[pool["Name"].isin(weights.keys())].reset_index(drop=True)
    if pool.empty:
        return [] if min_req == 0 else None

    count = len(pool)
    if count < min_req:
        return None

    max_req = min(max_req, count)
    if min_req > max_req:
        return None

    k = random.randint(min_req, max_req)

    w = [weights.get(n, 0.0) for n in pool["Name"]]
    if sum(w) == 0:
        w = [1.0] * len(w)

    chosen = []
    remaining = list(range(len(pool)))
    rem_w = [w[i] for i in remaining]

    for _ in range(k):
        total = sum(rem_w)
        if total == 0:
            probs = [1/len(remaining)] * len(remaining)
        else:
            probs = [x/total for x in rem_w]

        r = random.random()
        acc = 0
        idx = 0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                idx = i
                break

        chosen_idx = remaining.pop(idx)
        rem_w.pop(idx)
        chosen.append(pool.iloc[chosen_idx])

    return chosen





# ================================================================
#                    LINEUP BUILDER (FINAL)
# ================================================================

def build_stack_lineup(df, pos_groups, primary_team, mini_rule, opp_map):
    required = STACK_REQUIRED.get(primary_team, [])
    stack_players = []

    # Required
    for name in required:
        r = df[(df["Name"] == name) & (df["TeamAbbrev"] == primary_team)]
        if r.empty:
            return None
        stack_players.append(r.iloc[0])

    # Optional sprinkles
    for name in sample_optional_players(primary_team):
        r = df[(df["Name"] == name) & (df["TeamAbbrev"] == primary_team)]
        if not r.empty:
            stack_players.append(r.iloc[0])

    # Runbacks
    rb = select_runbacks_for_stack(primary_team, df)
    if rb is None:
        return None
    stack_players += rb

    # DST sprinkle
    if STACK_INCLUDE_DST.get(primary_team, False):
        if random.random() < STACK_DST_PERCENT.get(primary_team, 0.0):
            dst = df[(df["Position"] == "DST") &
                     (df["TeamAbbrev"] == primary_team)]
            if not dst.empty:
                stack_players.append(dst.iloc[0])

    used_ids = set(p.ID for p in stack_players)

    # Mini-stack
    corr = []
    if mini_rule:
        pick = pick_mini_stack_players(mini_rule, df, used_ids)
        if pick is None:
            return None
        for p in pick:
            corr.append(p)
            used_ids.add(p.ID)

    base = stack_players + corr

    # Position lists
    Q = [p for p in base if p.Position == "QB"]
    R = [p for p in base if p.Position == "RB"]
    W = [p for p in base if p.Position == "WR"]
    T = [p for p in base if p.Position == "TE"]
    D = [p for p in base if p.Position == "DST"]

    # Enforce stack min/max
    min_p, max_p = STACK_MIN_MAX.get(primary_team, (2, 5))
    count_primary = sum(1 for p in base if p.TeamAbbrev == primary_team)
    if not (min_p <= count_primary <= max_p):
        return None

    # Start with a copy of df
    df2 = df.copy()

    # Mini exclusivity
    if mini_rule and corr:
        ids = set(p.ID for p in corr)
        if mini_rule["type"] == "same_team":
            t = mini_rule["team"]
            df2 = df2[(df2["TeamAbbrev"] != t) | (df2["ID"].isin(ids))]
        else:
            t1, t2 = mini_rule["team1"], mini_rule["team2"]
            df2 = df2[
                ((df2["TeamAbbrev"] != t1) &
                 (df2["TeamAbbrev"] != t2)) |
                (df2["ID"].isin(ids))
            ]

    stack_set = set(STACK_TEAMS)
    runback_set = set(STACK_RUNBACK_TEAMS.values())

    filler = df2[
        (~df2["TeamAbbrev"].isin(stack_set)) &
        (~df2["TeamAbbrev"].isin(runback_set))
    ]

    def pool(pos):
        x = filler[filler["Position"] == pos]
        return x[~x.ID.isin(used_ids)]

    # Fill positions
    if Q:
        qb = Q[0]
    else:
        p = pool("QB")
        if p.empty:
            return None
        qb = p.sample(1).iloc[0]
    used_ids.add(qb.ID)

    while len(R) < 2:
        p = pool("RB")
        if p.empty:
            return None
        row = p.sample(1).iloc[0]
        R.append(row)
        used_ids.add(row.ID)

    while len(W) < 3:
        p = pool("WR")
        if p.empty:
            return None
        row = p.sample(1).iloc[0]
        W.append(row)
        used_ids.add(row.ID)

    if T:
        te = T[0]
    else:
        p = pool("TE")
        if p.empty:
            return None
        te = p.sample(1).iloc[0]
    used_ids.add(te.ID)

    if D:
        dst = D[0]
    else:
        p = pool("DST")
        if p.empty:
            return None
        dst = p.sample(1).iloc[0]
    used_ids.add(dst.ID)

    flex = filler[
        (filler.Position.isin(FLEX_ELIGIBLE)) &
        (~filler.ID.isin(used_ids))
    ]
    if flex.empty:
        return None
    flex = flex.sample(1).iloc[0]

    # Check salary
    lineup = [
        {"Slot": "QB", "Player": qb},
        {"Slot": "RB1", "Player": R[0]},
        {"Slot": "RB2", "Player": R[1]},
        {"Slot": "WR1", "Player": W[0]},
        {"Slot": "WR2", "Player": W[1]},
        {"Slot": "WR3", "Player": W[2]},
        {"Slot": "TE", "Player": te},
        {"Slot": "DST", "Player": dst},
        {"Slot": "FLEX", "Player": flex},
    ]

    total = sum(p["Player"].Salary for p in lineup)
    if not (MIN_SALARY <= total <= SALARY_CAP):
        return None

    return lineup




# ================================================================
#                        STREAMLIT UI
# ================================================================

def run_app():
    global NUM_LINEUPS, SALARY_CAP, MIN_SALARY, RANDOM_SEED
    global STACK_TEAMS, STACK_EXPOSURES, STACK_REQUIRED, STACK_OPTIONAL, STACK_MIN_MAX
    global STACK_RUNBACK_TEAMS, STACK_RUNBACKS, STACK_RUNBACK_MIN_MAX
    global STACK_INCLUDE_DST, STACK_DST_PERCENT
    global MINI_STACKS
    global TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE

    st.title("🏈 Main Slate DFS Lineup Builder — Stacks + Runbacks + Mini-stacks")

    # ------------------- CSV UPLOAD -------------------
    uploaded = st.file_uploader("Upload a DKSalaries.csv file", type=["csv"])
    if not uploaded:
        st.info("Please upload a **DKSalaries.csv** file to continue.")
        return

    df_raw = load_player_pool(uploaded)

    df_raw["Name"] = df_raw["Name"].astype(str)
    df_raw["Position"] = df_raw["Position"].astype(str)
    df_raw["TeamAbbrev"] = df_raw["TeamAbbrev"].astype(str)

    all_teams = sorted(df_raw["TeamAbbrev"].unique().tolist())
    opponent_map = extract_opponents(df_raw)

    # ------------------- SIDEBAR SETTINGS -------------------
    st.sidebar.header("Global Build Settings")

    num_lineups = st.sidebar.number_input(
        "Number of lineups", min_value=1, max_value=200, value=NUM_LINEUPS,
    )
    salary_cap = st.sidebar.number_input(
        "Salary cap", min_value=10000, max_value=50000, value=SALARY_CAP,
    )
    min_salary = st.sidebar.number_input(
        "Minimum salary", min_value=0, max_value=salary_cap, value=MIN_SALARY,
    )
    seed = st.sidebar.number_input(
        "Random seed (-1 for random)", value=RANDOM_SEED,
    )

    # ------------------- MAIN TABS -------------------
    tab_filters, tab_stacks, tab_runbacks, tab_minis, tab_build = st.tabs(
        ["Global Filters", "Stack Teams", "Run-backs", "Mini-stacks", "Build Lineups"]
    )

    # ================================================================
    #                       GLOBAL FILTERS
    # ================================================================
    with tab_filters:
        st.subheader("Global Player/Team Filtering")

        TEAM_FILTER_MODE.clear()
        TEAM_FILTER_KEEP.clear()
        TEAM_FILTER_EXCLUDE.clear()

        for t in all_teams:
            with st.expander(f"Team: {t}"):
                mode = st.radio(
                    f"Filter Mode for {t}:",
                    ["none", "remove_team", "keep_only", "exclude_only"],
                    key=f"filter_mode_{t}"
                )
                TEAM_FILTER_MODE[t] = mode

                if mode == "keep_only":
                    TEAM_FILTER_KEEP[t] = st.multiselect(
                        f"Players to KEEP from {t}:",
                        df_raw[df_raw["TeamAbbrev"] == t]["Name"].unique().tolist(),
                        key=f"keep_{t}"
                    )
                elif mode == "exclude_only":
                    TEAM_FILTER_EXCLUDE[t] = st.multiselect(
                        f"Players to EXCLUDE from {t}:",
                        df_raw[df_raw["TeamAbbrev"] == t]["Name"].unique().tolist(),
                        key=f"exclude_{t}"
                    )
                else:
                    TEAM_FILTER_KEEP[t] = []
                    TEAM_FILTER_EXCLUDE[t] = []

        df_filtered = apply_global_team_filters(
            df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
        )
        filtered_teams = sorted(df_filtered["TeamAbbrev"].unique().tolist())

        st.success(f"Players remaining: {len(df_filtered)}")

    # ================================================================
    #                         STACK TEAMS
    # ================================================================
    with tab_stacks:
        st.subheader("Primary Stack Teams")

        STACK_TEAMS = st.multiselect(
            "Select stack teams:", filtered_teams, default=[]
        )

        STACK_EXPOSURES.clear()
        STACK_REQUIRED.clear()
        STACK_OPTIONAL.clear()
        STACK_MIN_MAX.clear()

        for team in STACK_TEAMS:
            with st.expander(f"Stack Rules for {team}"):

                # Dynamic max exposure
                already_used = sum(
                    STACK_EXPOSURES.get(t, 0.0)
                    for t in STACK_TEAMS if t != team
                )
                remaining = max(0.0, 1.0 - already_used)
                curr = STACK_EXPOSURES.get(team, 0.0) * 100

                exp = st.slider(
                    f"{team} Exposure (%)",
                    0.0, remaining * 100.0, curr,
                    1.0, key=f"exposure_{team}"
                )
                STACK_EXPOSURES[team] = exp / 100.0

                # Min/Max
                c1, c2 = st.columns(2)
                mn = c1.number_input(
                    f"Min players from {team}", min_value=1, max_value=9, value=2,
                    key=f"min_{team}"
                )
                mx = c2.number_input(
                    f"Max players from {team}", min_value=mn, max_value=9, value=5,
                    key=f"max_{team}"
                )
                STACK_MIN_MAX[team] = (mn, mx)

                team_players = df_filtered[df_filtered["TeamAbbrev"] == team]["Name"].tolist()

                required = st.multiselect(
                    f"Required players:", team_players, key=f"req_{team}"
                )
                STACK_REQUIRED[team] = required

                optional = st.multiselect(
                    f"Optional sprinkle players:", team_players, key=f"opt_{team}"
                )

                om = {}
                for p in optional:
                    pct = st.slider(
                        f"{p} sprinkle chance (%)", 0.0, 100.0, 0.0, 1.0,
                        key=f"pct_opt_{team}_{p}"
                    )
                    om[p] = pct / 100.0
                STACK_OPTIONAL[team] = om

    # ================================================================
    #                         RUNBACKS (patched)
    # ================================================================
    with tab_runbacks:
        st.subheader("Run-back Settings (Mode B)")

        STACK_RUNBACK_TEAMS.clear()
        STACK_RUNBACKS.clear()
        STACK_RUNBACK_MIN_MAX.clear()
        STACK_INCLUDE_DST.clear()
        STACK_DST_PERCENT.clear()

        for team in STACK_TEAMS:
            opp = opponent_map.get(team, "")

            with st.expander(f"Runbacks for {team}"):

                if not opp or opp not in filtered_teams:
                    STACK_RUNBACK_TEAMS[team] = ""
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 0)
                    st.info(f"No valid opponent for {team}.")
                    continue

                STACK_RUNBACK_TEAMS[team] = opp
                st.write(f"Opponent: **{opp}**")

                # 1) Compute locked positions for stack team
                locked_QB = locked_RB = locked_WR = locked_TE = locked_DST = 0

                req = STACK_REQUIRED.get(team, [])
                req_rows = df_filtered[
                    (df_filtered["TeamAbbrev"] == team) &
                    (df_filtered["Name"].isin(req))
                ]
                locked_QB += (req_rows["Position"] == "QB").sum()
                locked_RB += (req_rows["Position"] == "RB").sum()
                locked_WR += (req_rows["Position"] == "WR").sum()
                locked_TE += (req_rows["Position"] == "TE").sum()
                locked_DST+= (req_rows["Position"] == "DST").sum()

                opt = STACK_OPTIONAL.get(team, {})
                always = [p for p, pct in opt.items() if pct >= 1.0]
                if always:
                    opt_rows = df_filtered[
                        (df_filtered["TeamAbbrev"] == team) &
                        (df_filtered["Name"].isin(always))
                    ]
                    locked_RB += (opt_rows["Position"] == "RB").sum()
                    locked_WR += (opt_rows["Position"] == "WR").sum()
                    locked_TE += (opt_rows["Position"] == "TE").sum()
                    locked_DST+= (opt_rows["Position"] == "DST").sum()

                # Convert to remaining slots
                remaining_RB = max(0, 3 - locked_RB)
                remaining_WR = max(0, 4 - locked_WR)
                remaining_TE = max(0, 2 - locked_TE)
                remaining_DST = max(0, 1 - locked_DST)

                # 2) Opponent pool
                opp_pool = df_filtered[
                    (df_filtered["TeamAbbrev"] == opp) &
                    (df_filtered["Position"] != "QB")
                ]

                if remaining_RB == 0:
                    opp_pool = opp_pool[opp_pool["Position"] != "RB"]
                if remaining_WR == 0:
                    opp_pool = opp_pool[opp_pool["Position"] != "WR"]
                if remaining_TE == 0:
                    opp_pool = opp_pool[opp_pool["Position"] != "TE"]
                if remaining_DST == 0:
                    opp_pool = opp_pool[opp_pool["Position"] != "DST"]

                opp_names = sorted(opp_pool["Name"].unique().tolist())

                if not opp_names:
                    st.info("No eligible opponent players given remaining positional slots.")
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 0)
                    continue

                # 3) Player selection with auto-clean
                prev = st.session_state.get(f"rbsel_{team}", [])
                cleaned = [p for p in prev if p in opp_names]

                rb_sel = st.multiselect(
                    f"Run-back players from {opp}:",
                    options=opp_names,
                    default=cleaned,
                    key=f"rbsel_{team}"
                )

                # 4) Min/Max (safe — no session_state writes)
                max_possible = len(rb_sel)

                mn = st.number_input(
                    f"Min run-backs",
                    min_value=0, max_value=max_possible,
                    value=min(st.session_state.get(f"rbmin_{team}", 0), max_possible),
                    key=f"rbmin_{team}"
                )
                mx = st.number_input(
                    f"Max run-backs",
                    min_value=mn, max_value=max_possible,
                    value=min(st.session_state.get(f"rbmax_{team}", 1), max_possible),
                    key=f"rbmax_{team}"
                )

                STACK_RUNBACK_MIN_MAX[team] = (mn, mx)

                # 5) Weight logic (Mode B)
                rb_map = {}
                for p in rb_sel:
                    slider_key = f"rbpct_{team}_{p}"
                    stored = st.session_state.get(slider_key, 0.0)

                    if mn == 1 and mx == 1:
                        used = sum(
                            rb_map.get(x, st.session_state.get(f"rbpct_{team}_{x}", 0)/100)
                            for x in rb_sel if x != p
                        )
                        remaining = max(0.0, 1.0 - used)
                        curr = min(stored, remaining * 100.0)

                        pct = st.slider(
                            f"{p} runback weight (%)",
                            0.0,
                            remaining * 100.0,
                            curr,
                            1.0,
                            key=slider_key
                        )
                        rb_map[p] = pct / 100.0
                    else:
                        pct = st.slider(
                            f"{p} runback weight (relative)",
                            0.0, 100.0,
                            stored,
                            1.0,
                            key=slider_key
                        )
                        rb_map[p] = pct / 100.0

                STACK_RUNBACKS[team] = rb_map

                # 6) Allow DST for stack team
                inc_dst = st.checkbox(f"Allow {team} DST sprinkle?", key=f"dstinc_{team}")
                STACK_INCLUDE_DST[team] = inc_dst

                dst_pct = st.slider(
                    f"{team} DST sprinkle chance (%)",
                    0.0, 100.0,
                    st.session_state.get(f"dstpct_{team}", 0.0),
                    1.0,
                    key=f"dstpct_{team}"
                )
                STACK_DST_PERCENT[team] = dst_pct / 100.0

    # ================================================================
    #                        MINI-STACK TAB
    # ================================================================
    # (unchanged — already validated cleanly)
    # ================================================================

    # ================================================================
    #                   BUILD LINEUPS TAB (unchanged)
    # ================================================================
    with tab_build:
        st.subheader("Generate Lineups")

        if st.button("🚀 Build Lineups"):
            NUM_LINEUPS = int(num_lineups)
            SALARY_CAP = int(salary_cap)
            MIN_SALARY = int(min_salary)

            if seed >= 0:
                random.seed(int(seed))

            if not STACK_TEAMS:
                st.info("Please select at least one stack team.")
                return

            if sum(STACK_EXPOSURES.values()) == 0:
                st.info("At least one stack team must have >0% exposure.")
                return

            df_final = apply_global_team_filters(
                df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
            )
            opponent_map_final = extract_opponents(df_final)
            pos_groups = position_split(df_final)

            # Determine lineups per stack
            stack_counts = {
                t: int(NUM_LINEUPS * STACK_EXPOSURES[t]) for t in STACK_TEAMS
            }

            # Build mini-stack state
            MINI_STACKS = []
            for rule in st.session_state.get("mini_rules", []):
                p = rule.get("exposure_pct", 0.0) / 100.0
                if p > 0:
                    if rule["type"] == "same_team":
                        if rule["team"] and rule.get("player1") and rule.get("player2"):
                            MINI_STACKS.append({
                                "type": "same_team",
                                "team": rule["team"],
                                "players": [rule["player1"], rule["player2"]],
                                "exposure": p,
                            })
                    else:
                        if rule["team1"] and rule["team2"] and rule.get("player1") and rule.get("player2"):
                            MINI_STACKS.append({
                                "type": "opposing_teams",
                                "team1": rule["team1"],
                                "team2": rule["team2"],
                                "players": [rule["player1"], rule["player2"]],
                                "exposure": p,
                            })

            mini_state = init_mini_stack_state(NUM_LINEUPS, MINI_STACKS)

            def pick_mini_rule(stack_team):
                for r in mini_state:
                    if r["remaining"] > 0 and mini_rule_applicable(
                        r, stack_team, STACK_TEAMS, STACK_RUNBACK_TEAMS
                    ):
                        return r
                return None

            # BUILD LOOP
            import time
            lineups = []
            used_keys = set()

            for team in STACK_TEAMS:
                target = stack_counts[team]
                built = 0
                st.info(f"Building {target} lineups for {team}…")
                prog = st.progress(0)
                start = time.time()

                while built < target:
                    attempts = 0
                    success = False

                    while attempts < MAX_ATTEMPTS_PER_LINEUP:
                        attempts += 1
                        m_rule = pick_mini_rule(team)

                        lu = build_stack_lineup(
                            df_final, pos_groups, team, m_rule, opponent_map_final
                        )
                        if lu is None:
                            continue

                        key = tuple(sorted(x["Player"].ID for x in lu))
                        if key in used_keys:
                            continue

                        used_keys.add(key)
                        lineups.append(lu)
                        built += 1
                        success = True
                        if m_rule:
                            m_rule["remaining"] -= 1
                        break

                    if not success:
                        st.info(
                            f"Could not find more unique lineups for {team} after "
                            f"{MAX_ATTEMPTS_PER_LINEUP} attempts."
                        )
                        break

                    prog.progress(built / max(target, 1))
                    elapsed = time.time() - start
                    if built > 0:
                        eta = (target - built) * (elapsed / built)
                        st.caption(f"{team}: {built}/{target} — ETA {eta:.1f}s")

                st.info(f"Completed {built}/{target} for {team}.")

            if not lineups:
                st.info("No lineups generated. Loosen constraints.")
                return

            st.success(f"Generated {len(lineups)} lineups!")

            # Convert to DataFrame
            def lineups_to_df(lineups):
                rows = []
                for i, lu in enumerate(lineups, start=1):
                    rec = {"LineupID": i}
                    total = 0
                    for slot in SLOT_ORDER:
                        p = next(x["Player"] for x in lu if x["Slot"] == slot)
                        rec[slot] = f"{p.Name} {p.ID}"
                        total += p.Salary
                    rec["Total Salary"] = total
                    rows.append(rec)
                return pd.DataFrame(rows)

            df_out = lineups_to_df(lineups)
            st.dataframe(df_out)

            st.download_button(
                "Download CSV",
                df_out.to_csv(index=False).encode("utf-8"),
                "DFS_Lineups.csv",
                "text/csv"
            )


if __name__ == "__main__":
    run_app()
