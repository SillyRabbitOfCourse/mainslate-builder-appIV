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
    # Ensure core columns are string where needed
    df_raw["Name"] = df_raw["Name"].astype(str)
    df_raw["Position"] = df_raw["Position"].astype(str)
    df_raw["TeamAbbrev"] = df_raw["TeamAbbrev"].astype(str)

    all_teams = sorted(df_raw["TeamAbbrev"].unique().tolist())
    opponent_map = extract_opponents(df_raw)

    # ------------------- SIDEBAR SETTINGS -------------------
    st.sidebar.header("Global Build Settings")

    num_lineups = st.sidebar.number_input(
        "Number of lineups",
        min_value=1,
        max_value=200,
        value=NUM_LINEUPS,
    )
    salary_cap = st.sidebar.number_input(
        "Salary cap",
        min_value=10000,
        max_value=50000,
        value=SALARY_CAP,
    )
    min_salary = st.sidebar.number_input(
        "Minimum salary",
        min_value=0,
        max_value=salary_cap,
        value=MIN_SALARY,
    )
    seed = st.sidebar.number_input(
        "Random seed (-1 for random)",
        value=RANDOM_SEED,
    )

    # ------------------- MAIN TABS -------------------
    tab_filters, tab_stacks, tab_runbacks, tab_minis, tab_build = st.tabs(
        [
            "Global Filters",
            "Stack Teams",
            "Run-backs",
            "Mini-stacks",
            "Build Lineups",
        ]
    )

    # ================================================================
    #                       GLOBAL TEAM FILTER TAB
    # ================================================================
    with tab_filters:
        st.subheader("Global Player/Team Filtering (applied BEFORE any stacks)")

        TEAM_FILTER_MODE.clear()
        TEAM_FILTER_KEEP.clear()
        TEAM_FILTER_EXCLUDE.clear()

        st.caption("""
        **Each team is independent.**  
        Filtering here happens *before* stack/runback/mini-stack selection.
        """)

        for t in all_teams:
            with st.expander(f"Team: {t}", expanded=False):
                mode = st.radio(
                    f"Filtering mode for {t}:",
                    ["none", "remove_team", "keep_only", "exclude_only"],
                    index=0,
                    key=f"filter_mode_{t}",
                )
                TEAM_FILTER_MODE[t] = mode

                if mode == "keep_only":
                    keep_list = st.multiselect(
                        f"Players to KEEP from {t}: (others removed)",
                        options=sorted(
                            df_raw[df_raw["TeamAbbrev"] == t]["Name"].unique().tolist()
                        ),
                        key=f"keep_{t}",
                    )
                    TEAM_FILTER_KEEP[t] = keep_list

                elif mode == "exclude_only":
                    exclude_list = st.multiselect(
                        f"Players to EXCLUDE from {t}:",
                        options=sorted(
                            df_raw[df_raw["TeamAbbrev"] == t]["Name"].unique().tolist()
                        ),
                        key=f"exclude_{t}",
                    )
                    TEAM_FILTER_EXCLUDE[t] = exclude_list

                else:
                    TEAM_FILTER_KEEP[t] = []
                    TEAM_FILTER_EXCLUDE[t] = []

        df_filtered = apply_global_team_filters(
            df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
        )
        filtered_teams = sorted(df_filtered["TeamAbbrev"].unique().tolist())
        st.success(f"Players after filtering: {len(df_filtered)}")

    # ================================================================
    #                           STACK TEAMS
    # ================================================================
    with tab_stacks:
        st.subheader("Primary Stack Teams (Offenses You Want To Build Around)")

        STACK_TEAMS = st.multiselect(
            "Select stack teams:",
            options=filtered_teams,
            default=[],
        )

        STACK_EXPOSURES.clear()
        STACK_REQUIRED.clear()
        STACK_OPTIONAL.clear()
        STACK_MIN_MAX.clear()

        if STACK_TEAMS:
            st.markdown("### Configure Stack Rules")

        for team in STACK_TEAMS:
            with st.expander(f"Stack Team: {team}", expanded=False):

                # Dynamic max exposure based on other teams
                already_used = sum(
                    STACK_EXPOSURES.get(t, 0.0) for t in STACK_TEAMS if t != team
                )
                remaining = max(0.0, 1.0 - already_used)
                current_val = STACK_EXPOSURES.get(team, 0.0) * 100.0
                if current_val > remaining * 100.0:
                    current_val = remaining * 100.0

                exp = st.slider(
                    f"{team} stack exposure (%)",
                    0.0,
                    remaining * 100.0,
                    current_val,
                    1.0,
                    key=f"exposure_{team}",
                )
                STACK_EXPOSURES[team] = exp / 100.0

                colA, colB = st.columns(2)
                with colA:
                    mn = st.number_input(
                        f"Min players from {team}",
                        min_value=1, max_value=9, value=2,
                        key=f"min_{team}"
                    )
                with colB:
                    mx = st.number_input(
                        f"Max players from {team}",
                        min_value=mn, max_value=9, value=5,
                        key=f"max_{team}"
                    )
                STACK_MIN_MAX[team] = (mn, mx)

                team_players = sorted(
                    df_filtered[df_filtered["TeamAbbrev"] == team]["Name"].unique().tolist()
                )
                required = st.multiselect(
                    f"Required players ({team}):",
                    options=team_players,
                    key=f"required_{team}",
                )
                STACK_REQUIRED[team] = required

                optional = st.multiselect(
                    f"Optional sprinkle players ({team}):",
                    options=team_players,
                    key=f"optional_{team}",
                )

                sprinkle_map = {}
                for p in optional:
                    pct = st.slider(
                        f"{p} sprinkle chance (%)",
                        0.0, 100.0, 0.0, 1.0,
                        key=f"opt_pct_{team}_{p}",
                    )
                    sprinkle_map[p] = pct / 100.0
                STACK_OPTIONAL[team] = sprinkle_map

        if STACK_TEAMS:
            st.write("**Expected lineups per stack:**")
            for t in STACK_TEAMS:
                exp = STACK_EXPOSURES.get(t, 0.0)
                expected_lineups = num_lineups * exp
                st.caption(
                    f"- {t}: {exp*100:.1f}% → ~{expected_lineups:.1f} of {num_lineups} lineups"
                )

    # ================================================================
    #                           RUN-BACKS
    # ================================================================
    with tab_runbacks:
        st.subheader("Run-back Settings (Mode B: weighted, min/max aware)")

        STACK_RUNBACK_TEAMS.clear()
        STACK_RUNBACKS.clear()
        STACK_RUNBACK_MIN_MAX.clear()
        STACK_INCLUDE_DST.clear()
        STACK_DST_PERCENT.clear()

        for team in STACK_TEAMS:
            opp = opponent_map.get(team, "")

            with st.expander(f"Run-backs for {team}", expanded=False):

                if not opp or opp not in filtered_teams:
                    st.info(f"No valid opponent for {team}. Run-backs disabled.")
                    STACK_RUNBACK_TEAMS[team] = ""
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 0)
                    continue

                STACK_RUNBACK_TEAMS[team] = opp
                st.write(f"**Opponent:** {opp}")

                # --------------------------------------------------------
                # 1) Compute locked positions for stack team (all slots)
                # --------------------------------------------------------
                locked_QB = locked_RB = locked_WR = locked_TE = locked_DST = 0

                req_names = STACK_REQUIRED.get(team, [])
                if req_names:
                    req_rows = df_filtered[
                        (df_filtered["TeamAbbrev"] == team) &
                        (df_filtered["Name"].isin(req_names))
                    ]
                    locked_QB  += (req_rows["Position"] == "QB").sum()
                    locked_RB  += (req_rows["Position"] == "RB").sum()
                    locked_WR  += (req_rows["Position"] == "WR").sum()
                    locked_TE  += (req_rows["Position"] == "TE").sum()
                    locked_DST += (req_rows["Position"] == "DST").sum()

                opt_map = STACK_OPTIONAL.get(team, {})
                always = [name for name, pct in opt_map.items() if pct >= 1.0]
                if always:
                    opt_rows = df_filtered[
                        (df_filtered["TeamAbbrev"] == team) &
                        (df_filtered["Name"].isin(always))
                    ]
                    locked_QB  += (opt_rows["Position"] == "QB").sum()
                    locked_RB  += (opt_rows["Position"] == "RB").sum()
                    locked_WR  += (opt_rows["Position"] == "WR").sum()
                    locked_TE  += (opt_rows["Position"] == "TE").sum()
                    locked_DST += (opt_rows["Position"] == "DST").sum()

                # DK max usage including FLEX
                remaining_QB  = max(0, 1 - locked_QB)
                remaining_RB  = max(0, 3 - locked_RB)   # 2 RB + 1 flex
                remaining_WR  = max(0, 4 - locked_WR)   # 3 WR + 1 flex
                remaining_TE  = max(0, 2 - locked_TE)   # 1 TE + 1 flex
                remaining_DST = max(0, 1 - locked_DST)

                # --------------------------------------------------------
                # 2) Base opponent pool (no QB) then prune impossible pos
                # --------------------------------------------------------
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
                    st.info(
                        f"No eligible non-QB opponent players from {opp} remain "
                        f"given locked {team} positions. Run-backs unavailable."
                    )
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 0)
                    continue

                # --------------------------------------------------------
                # 3) Player selection (auto-drop now-impossible players)
                # --------------------------------------------------------
                existing_rb_sel = st.session_state.get(f"rbsel_{team}", [])
                cleaned_sel = [p for p in existing_rb_sel if p in opp_names]

                rb_sel = st.multiselect(
                    f"Run-back players from {opp}:",
                    options=opp_names,
                    default=cleaned_sel,
                    key=f"rbsel_{team}",
                )

                # Drop any weights for players no longer selected
                rb_map: Dict[str, float] = {}
                stored_weights = {
                    k.split(f"rbpct_{team}_")[-1]: v / 100.0
                    for k, v in st.session_state.items()
                    if k.startswith(f"rbpct_{team}_")
                }

                # --------------------------------------------------------
                # 4) Min/Max settings with automatic caps
                # --------------------------------------------------------
                colA, colB = st.columns(2)
                with colA:
                    mn = st.number_input(
                        f"Min run-backs for {team}",
                        min_value=0, max_value=3, value=0,
                        key=f"rbmin_{team}"
                    )
                with colB:
                    mx = st.number_input(
                        f"Max run-backs for {team}",
                        min_value=mn, max_value=3, value=1,
                        key=f"rbmax_{team}"
                    )

                # Cap by number of selectable players
                max_possible = len(rb_sel)

                if max_possible == 0:
                    mn = 0
                    mx = 0
                    st.session_state[f"rbmin_{team}"] = 0
                    st.session_state[f"rbmax_{team}"] = 0
                else:
                    if mn > max_possible:
                        mn = max_possible
                        st.session_state[f"rbmin_{team}"] = mn
                    if mx > max_possible:
                        mx = max_possible
                        st.session_state[f"rbmax_{team}"] = mx

                STACK_RUNBACK_MIN_MAX[team] = (mn, mx)

                # --------------------------------------------------------
                # 5) Weights: dynamic caps when min=1 & max=1
                # --------------------------------------------------------
                for p in rb_sel:
                    slider_key = f"rbpct_{team}_{p}"
                    if mn == 1 and mx == 1:
                        # We enforce sum(weights) <= 1.0
                        already_used = sum(
                            rb_map.get(name, stored_weights.get(name, 0.0))
                            for name in rb_sel
                            if name != p
                        )
                        remaining = max(0.0, 1.0 - already_used)
                        current_val = st.session_state.get(slider_key, 0.0)
                        if current_val > remaining * 100.0:
                            current_val = remaining * 100.0

                        pct = st.slider(
                            f"{p} run-back weight (%)",
                            0.0,
                            remaining * 100.0,
                            current_val,
                            1.0,
                            key=slider_key,
                        )
                        rb_map[p] = pct / 100.0
                    else:
                        pct = st.slider(
                            f"{p} run-back weight (relative, not absolute %) for {team} stacks",
                            0.0, 100.0,
                            st.session_state.get(slider_key, 0.0),
                            1.0,
                            key=slider_key,
                        )
                        rb_map[p] = pct / 100.0

                STACK_RUNBACKS[team] = rb_map

                # --------------------------------------------------------
                # 6) DST sprinkle for STACK TEAM
                # --------------------------------------------------------
                inc_dst = st.checkbox(
                    f"Allow {team} DST in some stacks?",
                    key=f"dstinc_{team}",
                )
                STACK_INCLUDE_DST[team] = inc_dst

                dst_pct = st.slider(
                    f"{team} DST chance (%)",
                    0.0, 100.0,
                    st.session_state.get(f"dstpct_{team}", 0.0),
                    1.0,
                    key=f"dstpct_{team}",
                )
                STACK_DST_PERCENT[team] = dst_pct / 100.0

    # ================================================================
    #                          MINI-STACKS
    # ================================================================
    with tab_minis:
        st.subheader("Mini-stacks (secondary correlations)")

        if "mini_rules" not in st.session_state:
            st.session_state["mini_rules"] = []

        mini_rules = st.session_state["mini_rules"]

        col_add1, col_add2 = st.columns(2)
        with col_add1:
            if st.button("➕ Add same-team mini-stack"):
                mini_rules.append({
                    "type": "same_team",
                    "team": "",
                    "player1": "",
                    "player2": "",
                    "exposure_pct": 0.0,
                })
        with col_add2:
            if st.button("➕ Add opposing-team mini-stack"):
                mini_rules.append({
                    "type": "opposing_teams",
                    "team1": "",
                    "team2": "",
                    "player1": "",
                    "player2": "",
                    "exposure_pct": 0.0,
                })

        remove_idx = []

        for i, rule in enumerate(mini_rules):
            with st.expander(f"Mini-Stack #{i+1} ({rule['type']})"):

                # Dynamic mini exposure cap (sum ≤ 100%)
                mini_used = sum(
                    (r.get("exposure_pct", 0.0) / 100.0)
                    for idx2, r in enumerate(mini_rules)
                    if idx2 != i
                )
                mini_remaining = max(0.0, 1.0 - mini_used)
                current_val = rule.get("exposure_pct", 0.0)
                if current_val > mini_remaining * 100.0:
                    current_val = mini_remaining * 100.0

                exp = st.slider(
                    "Exposure (%)",
                    0.0,
                    mini_remaining * 100.0,
                    current_val,
                    1.0,
                    key=f"mini_exp_{i}"
                )
                rule["exposure_pct"] = exp

                if st.button("Delete", key=f"mini_del_{i}"):
                    remove_idx.append(i)
                    continue

                if rule["type"] == "same_team":
                    rule["team"] = st.selectbox(
                        "Team:",
                        options=[""] + filtered_teams,
                        index=([""] + filtered_teams).index(rule["team"])
                        if rule["team"] in filtered_teams else 0,
                        key=f"mini_team_same_{i}",
                    )

                    if rule["team"]:
                        team_players = df_filtered[
                            df_filtered["TeamAbbrev"] == rule["team"]
                        ]["Name"].tolist()

                        rule["player1"] = st.selectbox(
                            "Player 1:",
                            [""] + team_players,
                            index=([""] + team_players).index(rule["player1"])
                            if rule["player1"] in team_players else 0,
                            key=f"mini_p1_same_{i}",
                        )
                        rule["player2"] = st.selectbox(
                            "Player 2:",
                            [""] + team_players,
                            index=([""] + team_players).index(rule["player2"])
                            if rule["player2"] in team_players else 0,
                            key=f"mini_p2_same_{i}",
                        )

                else:  # opposing_teams
                    rule["team1"] = st.selectbox(
                        "Team 1:",
                        options=[""] + filtered_teams,
                        index=([""] + filtered_teams).index(rule["team1"])
                        if rule["team1"] in filtered_teams else 0,
                        key=f"mini_team1_{i}",
                    )

                    if rule["team1"] in opponent_map:
                        rule["team2"] = opponent_map[rule["team1"]]
                    else:
                        rule["team2"] = ""

                    st.write(f"Team 2 (Opponent): **{rule['team2']}**")

                    p1_list = df_filtered[
                        df_filtered["TeamAbbrev"] == rule["team1"]
                    ]["Name"].tolist() if rule["team1"] else []

                    p2_list = df_filtered[
                        df_filtered["TeamAbbrev"] == rule["team2"]
                    ]["Name"].tolist() if rule["team2"] else []

                    rule["player1"] = st.selectbox(
                        "Player from Team 1:",
                        [""] + p1_list,
                        index=([""] + p1_list).index(rule["player1"])
                        if rule["player1"] in p1_list else 0,
                        key=f"mini_p1_opp_{i}",
                    )
                    rule["player2"] = st.selectbox(
                        "Player from Team 2:",
                        [""] + p2_list,
                        index=([""] + p2_list).index(rule["player2"])
                        if rule["player2"] in p2_list else 0,
                        key=f"mini_p2_opp_{i}",
                    )

        if remove_idx:
            st.session_state["mini_rules"] = [
                r for idx2, r in enumerate(mini_rules) if idx2 not in remove_idx
            ]

        if st.session_state["mini_rules"]:
            st.write("**Expected mini-stack usage (lineups):**")
            for idx, rule in enumerate(st.session_state["mini_rules"]):
                pct = rule.get("exposure_pct", 0.0) / 100.0
                expected_lineups = num_lineups * pct
                label = f"Mini #{idx+1} ({rule['type']})"
                st.caption(
                    f"- {label}: {pct*100:.1f}% → ~{expected_lineups:.1f} of {num_lineups} lineups"
                )

    # ================================================================
    #                         BUILD LINEUPS TAB
    # ================================================================
    with tab_build:
        st.subheader("Generate Lineups")

        if STACK_TEAMS:
            st.write("### Configuration Summary:")
            st.write(f"Primary stack teams: **{', '.join(STACK_TEAMS)}**")

        if st.button("🚀 Build Lineups"):
            NUM_LINEUPS = int(num_lineups)
            SALARY_CAP = int(salary_cap)
            MIN_SALARY = int(min_salary)
            RANDOM_SEED = None if seed < 0 else int(seed)

            if RANDOM_SEED is not None:
                random.seed(RANDOM_SEED)

            if not STACK_TEAMS:
                st.info("Select at least one stack team before building.")
                return

            if sum(STACK_EXPOSURES.get(t, 0.0) for t in STACK_TEAMS) == 0.0:
                st.info("All stack exposures are 0%. Increase at least one team.")
                return

            df_final = apply_global_team_filters(
                df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
            )
            opponent_map_final = extract_opponents(df_final)
            pos_groups = position_split(df_final)

            stack_counts = {
                team: int(NUM_LINEUPS * STACK_EXPOSURES.get(team, 0.0))
                for team in STACK_TEAMS
            }
            st.write("### Planned Lineups Per Stack:")
            st.json(stack_counts)

            MINI_STACKS = []
            for rule in st.session_state.get("mini_rules", []):
                pct = rule.get("exposure_pct", 0.0) / 100.0
                if pct <= 0:
                    continue
                if rule["type"] == "same_team":
                    if rule["team"] and rule["player1"] and rule["player2"]:
                        MINI_STACKS.append({
                            "type": "same_team",
                            "team": rule["team"],
                            "players": [rule["player1"], rule["player2"]],
                            "exposure": pct,
                        })
                else:
                    if rule["team1"] and rule["team2"] and rule["player1"] and rule["player2"]:
                        MINI_STACKS.append({
                            "type": "opposing_teams",
                            "team1": rule["team1"],
                            "team2": rule["team2"],
                            "players": [rule["player1"], rule["player2"]],
                            "exposure": pct,
                        })

            mini_state = init_mini_stack_state(NUM_LINEUPS, MINI_STACKS)

            def pick_mini_rule(stack_team):
                for r in mini_state:
                    if r["remaining"] <= 0:
                        continue
                    if mini_rule_applicable(r, stack_team, STACK_TEAMS, STACK_RUNBACK_TEAMS):
                        return r
                return None

            import time

            lineups = []
            used_keys = set()

            for team in STACK_TEAMS:
                target = stack_counts.get(team, 0)
                built = 0

                st.info(f"Building {target} lineups for team {team}...")
                progress = st.progress(0)
                start = time.time()

                while built < target:
                    attempts = 0
                    success = False

                    while attempts < MAX_ATTEMPTS_PER_LINEUP:
                        attempts += 1
                        m_rule = pick_mini_rule(team)

                        lu = build_stack_lineup(
                            df_final,
                            pos_groups,
                            team,
                            m_rule,
                            opponent_map_final,
                        )
                        if lu is None:
                            continue

                        key = tuple(sorted(p["Player"].ID for p in lu))
                        if key in used_keys:
                            continue

                        used_keys.add(key)
                        lineups.append(lu)
                        built += 1
                        success = True

                        if m_rule is not None and "remaining" in m_rule:
                            m_rule["remaining"] -= 1
                        break

                    if not success:
                        st.info(
                            f"Stopped building for {team}: could not find another distinct lineup "
                            f"after {MAX_ATTEMPTS_PER_LINEUP} attempts. "
                            f"Current constraints may be extremely tight."
                        )
                        break

                    progress.progress(built / max(target, 1))

                    elapsed = time.time() - start
                    if built > 0:
                        rate = elapsed / built
                        eta = (target - built) * rate
                        st.caption(f"{team}: built {built}/{target} — ETA ~{eta:.1f} seconds")

                st.info(f"Finished: built {built}/{target} lineups for {team}.")

            if not lineups:
                st.info("No lineups were generated. Loosen constraints and try again.")
                return

            st.success(f"Successfully generated {len(lineups)} lineups!")

            def lineups_to_df(lineups):
                rows = []
                for i, lu in enumerate(lineups, start=1):
                    rec = {"LineupID": i}
                    total = 0
                    for slot in SLOT_ORDER:
                        p = next(item["Player"] for item in lu if item["Slot"] == slot)
                        rec[slot] = f"{p.Name} {p.ID}"
                        total += p.Salary
                    rec["Total Salary"] = total
                    rows.append(rec)
                return pd.DataFrame(rows)

            df_out = lineups_to_df(lineups)
            st.dataframe(df_out)

            st.download_button(
                label="Download Lineups CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="DFS_Lineups.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    run_app()
