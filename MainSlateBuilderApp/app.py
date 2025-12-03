# ================================================================
#                    IMPORTS & GLOBAL CONFIG
# ================================================================

import random
from typing import List, Dict, Tuple
import pandas as pd
import streamlit as st
import re

# These will be overwritten by Streamlit UI
NUM_LINEUPS = 40
SALARY_CAP = 50000
MIN_SALARY = 49000          # start near DK cap
RANDOM_SEED = 42

SLOT_ORDER = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
FLEX_ELIGIBLE = {"RB", "WR", "TE"}

# Global config objects populated by UI
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

# EXCLUSION RULES (global team filter)
TEAM_FILTER_MODE: Dict[str, str] = {}              # "none", "remove_team", "keep_only", "exclude_only"
TEAM_FILTER_KEEP: Dict[str, List[str]] = {}
TEAM_FILTER_EXCLUDE: Dict[str, List[str]] = {}

MAX_ATTEMPTS_PER_LINEUP = 20000

# ================================================================
#                          DATA LOADING
# ================================================================

def load_player_pool(source) -> pd.DataFrame:
    """Load DKSalaries.csv and clean column names."""
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]

    # Normalize data types
    df["Salary"] = df["Salary"].astype(int)
    df["Name"] = df["Name"].astype(str)
    df["Position"] = df["Position"].astype(str)
    df["TeamAbbrev"] = df["TeamAbbrev"].astype(str)

    return df

# ================================================================
#                     OPPONENT EXTRACTION
# ================================================================

def extract_opponents(df: pd.DataFrame) -> Dict[str, str]:
    """
    Parse the Game Info column and build a mapping: TEAM -> OPPONENT_TEAM
    Example:
        IND -> JAX, JAX -> IND
    """
    matchup_map = {}
    for _, row in df.iterrows():
        info = row["Game Info"]
        match = re.match(r"([A-Z]+)@([A-Z]+)", info)
        if match:
            away, home = match.group(1), match.group(2)
            matchup_map[away] = home
            matchup_map[home] = away
    return matchup_map

# ================================================================
#                    GLOBAL PLAYER POOL FILTERING
# ================================================================

def apply_global_team_filters(
    df: pd.DataFrame,
    team_modes: Dict[str, str],
    keep_map: Dict[str, List[str]],
    exclude_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Apply per-team filtering BEFORE any stack, run-back, or mini-stack UI.
    """
    filtered_df = df.copy()

    for team, mode in team_modes.items():
        if mode == "none":
            continue

        team_rows = filtered_df["TeamAbbrev"] == team

        if mode == "remove_team":
            filtered_df = filtered_df[~team_rows]
            continue

        if mode == "keep_only":
            keep_list = keep_map.get(team, [])
            filtered_df = filtered_df[
                ~team_rows | filtered_df["Name"].isin(keep_list)
            ]
            continue

        if mode == "exclude_only":
            exclude_list = exclude_map.get(team, [])
            filtered_df = filtered_df[
                ~team_rows | ~filtered_df["Name"].isin(exclude_list)
            ]
            continue

    return filtered_df.reset_index(drop=True)





# ================================================================
#                   STACK + SPRINKLE SELECTION LOGIC
# ================================================================

def sample_optional_players(team: str) -> List[str]:
    """Randomly select optional sprinkle players for a stack team."""
    chosen = []
    for player_name, pct in STACK_OPTIONAL.get(team, {}).items():
        if random.random() < pct:
            chosen.append(player_name)
    return chosen

# ================================================================
#                           RUN-BACK HELPERS
# ================================================================

def get_runback_pool(df: pd.DataFrame, opponent_team: str) -> pd.DataFrame:
    """
    Returns all *eligible* run-back players for the opponent team.
    QBs are automatically filtered OUT here.
    """
    if not opponent_team:
        return pd.DataFrame()

    return df[
        (df["TeamAbbrev"] == opponent_team) &
        (df["Position"] != "QB")
    ].reset_index(drop=True)

# ================================================================
#                  POSITION-SPECIFIC DATA GROUPS
# ================================================================

def position_split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    groups = {}
    for pos, group in df.groupby("Position"):
        groups[pos] = group.reset_index(drop=True)
    return groups

# ================================================================
#                         MINI-STACK SYSTEM
# ================================================================

def init_mini_stack_state(num_lineups: int, mini_stacks: List[Dict]) -> List[Dict]:
    """
    Convert MINI_STACKS configuration into a mutable state:
       rule["remaining"] = number of lineups this rule
       should apply to.
    """
    rules = []
    for rule in mini_stacks:
        target = int(num_lineups * rule.get("exposure", 0.0))
        new_rule = dict(rule)
        new_rule["remaining"] = target
        rules.append(new_rule)
    return rules

def mini_rule_applicable_to_team(
    rule: Dict,
    primary_team: str,
    stack_teams: List[str],
    runback_map: Dict[str, str],
) -> bool:
    """
    A mini-stack rule can only activate if:
    - It does not involve the primary stack team
    - It does not involve ANY stack team
    - It does not involve ANY run-back team
    """
    stack_set = set(stack_teams)
    runback_set = set(runback_map.values())

    if rule["type"] == "same_team":
        team = rule["team"]
        if team in stack_set or team in runback_set or team == primary_team:
            return False
        return True

    if rule["type"] == "opposing_teams":
        t1 = rule["team1"]
        t2 = rule["team2"]
        if t1 in stack_set or t2 in stack_set:
            return False
        if t1 in runback_set or t2 in runback_set:
            return False
        if t1 == primary_team or t2 == primary_team:
            return False
        return True

    return False




# ================================================================
#                 MINI-STACK PLAYER PICKING (PLAYER-BASED)
# ================================================================

def pick_mini_stack_players(
    rule: Dict,
    df: pd.DataFrame,
    used_ids: set,
    stack_teams: List[str],
    runback_map: Dict[str, str],
):
    """
    Select exact players specified in the mini-stack UI.
    Returns a list of 2 pd.Series or None.
    """

    if rule["type"] == "same_team":
        team = rule["team"]
        p1_name, p2_name = rule["players"]

        p1 = df[(df["Name"] == p1_name) & (df["TeamAbbrev"] == team)]
        p2 = df[(df["Name"] == p2_name) & (df["TeamAbbrev"] == team)]
        if p1.empty or p2.empty:
            return None

        p1, p2 = p1.iloc[0], p2.iloc[0]
        if p1.ID in used_ids or p2.ID in used_ids:
            return None

        return [p1, p2]

    elif rule["type"] == "opposing_teams":
        t1, t2 = rule["team1"], rule["team2"]
        p1_name, p2_name = rule["players"]

        p1 = df[(df["Name"] == p1_name) & (df["TeamAbbrev"] == t1)]
        p2 = df[(df["Name"] == p2_name) & (df["TeamAbbrev"] == t2)]
        if p1.empty or p2.empty:
            return None

        p1, p2 = p1.iloc[0], p2.iloc[0]
        if p1.ID in used_ids or p2.ID in used_ids:
            return None

        return [p1, p2]

    return None

# ================================================================
#               RUN-BACK SELECTION (MODE B: WEIGHTED)
# ================================================================

def select_runbacks_for_stack(team: str, df: pd.DataFrame) -> List[pd.Series] | None:
    """
    Mode B:
      - STACK_RUNBACKS[team] stores weights per player (0–1 floats)
      - STACK_RUNBACK_MIN_MAX[team] gives (min, max)
      - We pick K in [min, max] (capped by available players)
      - Then sample K players WITHOUT replacement, weighted by these weights.
    """

    opp = STACK_RUNBACK_TEAMS.get(team, "")
    min_req, max_req = STACK_RUNBACK_MIN_MAX.get(team, (0, 999))

    if not opp:
        return [] if min_req == 0 else None

    pool = get_runback_pool(df, opp)
    if pool.empty:
        return [] if min_req == 0 else None

    weight_map = STACK_RUNBACKS.get(team, {})
    if not weight_map:
        return [] if min_req == 0 else None

    # Filter pool to only selected run-back players
    pool = pool[pool["Name"].isin(weight_map.keys())].reset_index(drop=True)
    if pool.empty:
        return [] if min_req == 0 else None

    available = len(pool)
    if available < min_req:
        return None  # impossible

    # Cap max_req by available
    max_k = min(max_req, available)
    if max_k == 0:
        return [] if min_req == 0 else None

    if min_req > max_k:
        return None

    # Choose K between min_req and max_k
    k = random.randint(min_req, max_k)

    # Build weights aligned to pool rows
    names = pool["Name"].tolist()
    weights = [max(weight_map.get(n, 0.0), 0.0) for n in names]

    # If all weights are zero, treat as uniform
    if sum(weights) <= 0:
        weights = [1.0] * len(names)

    # Weighted sample WITHOUT replacement
    chosen_rows = []
    remaining_idx = list(range(len(names)))
    remaining_weights = [weights[i] for i in remaining_idx]

    for _ in range(k):
        total_w = sum(remaining_weights)
        if total_w <= 0:
            probs = [1.0 / len(remaining_idx)] * len(remaining_idx)
        else:
            probs = [w / total_w for w in remaining_weights]

        r = random.random()
        cum = 0.0
        chosen_pos = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                chosen_pos = i
                break

        idx = remaining_idx.pop(chosen_pos)
        remaining_weights.pop(chosen_pos)
        chosen_rows.append(pool.iloc[idx])

    return chosen_rows






# ================================================================
#                         LINEUP BUILDER
# ================================================================

def build_stack_lineup(
    df: pd.DataFrame,
    pos_groups: Dict[str, pd.DataFrame],
    primary_team: str,
    mini_rule: Dict | None,
    opponent_map: Dict[str, str],
) -> List[Dict] | None:
    """
    Build one valid lineup for the specified stack team.
    Applies:
      - Stack-required players
      - Optional sprinkle players
      - Run-backs (Mode B)
      - Mini-stacks (if allowed)
      - Team isolation rules
      - Salary cap checks
      - FLEX must be RB/WR/TE (no DST)
    """

    # PRIMARY STACK REQUIRED PLAYERS
    required_list = STACK_REQUIRED.get(primary_team, [])
    stack_players: List[pd.Series] = []

    for name in required_list:
        row = df[(df["Name"] == name) & (df["TeamAbbrev"] == primary_team)]
        if row.empty:
            return None
        stack_players.append(row.iloc[0])

    # OPTIONAL SPRINKLES
    sprinkle_names = sample_optional_players(primary_team)
    for name in sprinkle_names:
        row = df[(df["Name"] == name) & (df["TeamAbbrev"] == primary_team)]
        if not row.empty:
            stack_players.append(row.iloc[0])

    # RUN-BACK HANDLING (Mode B)
    runbacks = select_runbacks_for_stack(primary_team, df)
    if runbacks is None:
        return None
    for rb in runbacks:
        stack_players.append(rb)

    # DST SPRINKLE (from primary team only)
    if STACK_INCLUDE_DST.get(primary_team, False):
        pct = STACK_DST_PERCENT.get(primary_team, 0.0)
        if random.random() < pct:
            dst = df[(df["Position"] == "DST") & (df["TeamAbbrev"] == primary_team)]
            if not dst.empty:
                stack_players.append(dst.iloc[0])

    used_ids = {p.ID for p in stack_players}
    player_objs = list(stack_players)

    # MINI-STACK PLAYERS
    corr_players = []
    if mini_rule is not None:
        pick = pick_mini_stack_players(
            mini_rule,
            df,
            used_ids,
            STACK_TEAMS,
            STACK_RUNBACK_TEAMS,
        )
        if pick is None:
            return None
        for p in pick:
            corr_players.append(p)
            used_ids.add(p.ID)

    # Combine everything before filling positions
    base_players = player_objs + corr_players

    # Identify which positions already have stack players
    stack_QBs = [p for p in base_players if p.Position == "QB"]
    stack_RBs = [p for p in base_players if p.Position == "RB"]
    stack_WRs = [p for p in base_players if p.Position == "WR"]
    stack_TEs = [p for p in base_players if p.Position == "TE"]
    stack_DSTs = [p for p in base_players if p.Position == "DST"]

    # Enforce primary team min/max
    min_p, max_p = STACK_MIN_MAX.get(primary_team, (2, 5))
    count_primary = sum(1 for p in base_players if p.TeamAbbrev == primary_team)
    if not (min_p <= count_primary <= max_p):
        return None

    # MINI-STACK TEAM EXCLUSIVITY (PER-LINEUP ONLY)
    if mini_rule is not None and corr_players:
        mini_ids = {p.ID for p in corr_players}
        if mini_rule["type"] == "same_team":
            t = mini_rule["team"]
            df = df[
                (df["TeamAbbrev"] != t) |
                (df["ID"].isin(mini_ids))
            ]
        elif mini_rule["type"] == "opposing_teams":
            t1 = mini_rule["team1"]
            t2 = mini_rule["team2"]
            df = df[
                ((df["TeamAbbrev"] != t1) & (df["TeamAbbrev"] != t2)) |
                (df["ID"].isin(mini_ids))
            ]

    # TEAM ISOLATION LOGIC
    stack_set = set(STACK_TEAMS)
    runback_set = set(STACK_RUNBACK_TEAMS.values())

    filler_df = df[
        (~df["TeamAbbrev"].isin(stack_set)) &
        (~df["TeamAbbrev"].isin(runback_set))
    ]

    def pool(pos: str):
        subset = filler_df[filler_df.Position == pos]
        return subset[~subset.ID.isin(used_ids)].reset_index(drop=True)

    # FILL QB
    if stack_QBs:
        qb = stack_QBs[0]
    else:
        p = pool("QB")
        if p.empty:
            return None
        qb = p.sample(1).iloc[0]
    used_ids.add(qb.ID)

    # FILL RB1 & RB2
    rbs = stack_RBs.copy()
    while len(rbs) < 2:
        p = pool("RB")
        if p.empty:
            return None
        row = p.sample(1).iloc[0]
        rbs.append(row)
        used_ids.add(row.ID)

    # FILL WR1, WR2, WR3
    wrs = stack_WRs.copy()
    while len(wrs) < 3:
        p = pool("WR")
        if p.empty:
            return None
        row = p.sample(1).iloc[0]
        wrs.append(row)
        used_ids.add(row.ID)

    # FILL TE
    if stack_TEs:
        te = stack_TEs[0]
    else:
        p = pool("TE")
        if p.empty:
            return None
        te = p.sample(1).iloc[0]
    used_ids.add(te.ID)

    # FILL DST
    if stack_DSTs:
        dst = stack_DSTs[0]
    else:
        p = pool("DST")
        if p.empty:
            return None
        dst = p.sample(1).iloc[0]
    used_ids.add(dst.ID)

    # FILL FLEX (NO DST)
    flex_pool = filler_df[
        (filler_df.Position.isin(FLEX_ELIGIBLE)) &
        (~filler_df.ID.isin(used_ids))
    ]
    if flex_pool.empty:
        return None
    flex = flex_pool.sample(1).iloc[0]

    lineup = [
        {"Slot": "QB",   "Player": qb},
        {"Slot": "RB1",  "Player": rbs[0]},
        {"Slot": "RB2",  "Player": rbs[1]},
        {"Slot": "WR1",  "Player": wrs[0]},
        {"Slot": "WR2",  "Player": wrs[1]},
        {"Slot": "WR3",  "Player": wrs[2]},
        {"Slot": "TE",   "Player": te},
        {"Slot": "DST",  "Player": dst},
        {"Slot": "FLEX", "Player": flex},
    ]

    total = sum([entry["Player"].Salary for entry in lineup])
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
                    st.warning(f"No valid opponent for {team}. Run-backs disabled.")
                    STACK_RUNBACK_TEAMS[team] = ""
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 999)
                    continue

                STACK_RUNBACK_TEAMS[team] = opp
                st.write(f"**Opponent:** {opp}")

                # WR slot awareness
                locked_wr_count = 0

                req_names = STACK_REQUIRED.get(team, [])
                if req_names:
                    req_rows = df_filtered[
                        (df_filtered["TeamAbbrev"] == team) &
                        (df_filtered["Name"].isin(req_names))
                    ]
                    locked_wr_count += (req_rows["Position"] == "WR").sum()

                opt_map = STACK_OPTIONAL.get(team, {})
                always_opt_names = [name for name, pct in opt_map.items() if pct >= 1.0]
                if always_opt_names:
                    opt_rows = df_filtered[
                        (df_filtered["TeamAbbrev"] == team) &
                        (df_filtered["Name"].isin(always_opt_names))
                    ]
                    locked_wr_count += (opt_rows["Position"] == "WR").sum()

                wr_slots_available = max(0, 4 - locked_wr_count)
                if wr_slots_available == 0:
                    st.warning(
                        f"All WR slots are effectively filled by locked {team} WRs. "
                        "WR run-backs from the opponent cannot be used in these stacks."
                    )

                opp_pool = df_filtered[
                    (df_filtered["TeamAbbrev"] == opp) &
                    (df_filtered["Position"] != "QB")
                ]
                if wr_slots_available == 0:
                    opp_pool = opp_pool[opp_pool["Position"] != "WR"]

                opp_names = sorted(opp_pool["Name"].unique().tolist())

                if not opp_names:
                    st.warning(
                        f"No eligible non-QB (and non-conflicting) players available from {opp} "
                        "for run-backs, given your current stack locks."
                    )
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 999)
                    continue

                rb_sel = st.multiselect(
                    f"Eligible run-back players from {opp}:",
                    options=opp_names,
                    key=f"rbsel_{team}",
                )

                rb_map: Dict[str, float] = {}

                # Temporary so we can compute dynamic caps
                # first pass: read session values, second pass: enforce caps
                # We'll do it in a single pass but compute "used" based on prior entries.

                # MIN/MAX SETTINGS
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

                if rb_sel and mn > len(rb_sel):
                    mn = len(rb_sel)
                    st.session_state[f"rbmin_{team}"] = mn
                if rb_sel and mx > len(rb_sel):
                    mx = len(rb_sel)
                    st.session_state[f"rbmax_{team}"] = mx

                STACK_RUNBACK_MIN_MAX[team] = (mn, mx)

                # Runback weights with dynamic caps IF min=1, max=1
                for p in rb_sel:
                    if mn == 1 and mx == 1:
                        # enforce sum(weights) <= 1.0
                        already_used = sum(
                            rb_map.get(name, 0.0) for name in rb_sel if name != p
                        )
                        remaining = max(0.0, 1.0 - already_used)
                        current_val = 0.0
                        if f"rbpct_{team}_{p}" in st.session_state:
                            current_val = st.session_state[f"rbpct_{team}_{p}"]
                        current_val = min(current_val, remaining * 100.0)

                        pct = st.slider(
                            f"{p} run-back weight (%)",
                            0.0,
                            remaining * 100.0,
                            current_val,
                            1.0,
                            key=f"rbpct_{team}_{p}",
                        )
                        rb_map[p] = pct / 100.0
                    else:
                        # general case: weight is relative, free in [0,100]
                        pct = st.slider(
                            f"{p} run-back weight (relative, not absolute %) for {team} stacks",
                            0.0, 100.0, 0.0, 1.0,
                            key=f"rbpct_{team}_{p}",
                        )
                        rb_map[p] = pct / 100.0

                STACK_RUNBACKS[team] = rb_map

                inc_dst = st.checkbox(
                    f"Allow {team} DST in some stacks?",
                    key=f"dstinc_{team}",
                )
                STACK_INCLUDE_DST[team] = inc_dst

                dst_pct = st.slider(
                    f"{team} DST chance (%)",
                    0.0, 100.0, 0.0, 1.0,
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

                # dynamic remaining mini exposure
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
                        key=f"mini_same_team_{i}",
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
                            key=f"mini_player1_same_{i}",
                        )

                        rule["player2"] = st.selectbox(
                            "Player 2:",
                            [""] + team_players,
                            index=([""] + team_players).index(rule["player2"])
                            if rule["player2"] in team_players else 0,
                            key=f"mini_player2_same_{i}",
                        )

                elif rule["type"] == "opposing_teams":
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
                        key=f"mini_player1_opp_{i}",
                    )

                    rule["player2"] = st.selectbox(
                        "Player from Team 2:",
                        [""] + p2_list,
                        index=([""] + p2_list).index(rule["player2"])
                        if rule["player2"] in p2_list else 0,
                        key=f"mini_player2_opp_{i}",
                    )

        if remove_idx:
            st.session_state["mini_rules"] = [
                r for idx, r in enumerate(mini_rules) if idx not in remove_idx
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
                st.error("Please select at least one stack team before building.")
                return

            if sum(STACK_EXPOSURES.get(t, 0.0) for t in STACK_TEAMS) == 0.0:
                st.error("All stack exposures are 0%. Please increase at least one team.")
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
                elif rule["type"] == "opposing_teams":
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
                    if mini_rule_applicable_to_team(
                        r, stack_team, STACK_TEAMS, STACK_RUNBACK_TEAMS
                    ):
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
                team_start = time.time()

                while built < target:
                    attempts_for_this_lineup = 0
                    success = False

                    while attempts_for_this_lineup < MAX_ATTEMPTS_PER_LINEUP:
                        attempts_for_this_lineup += 1

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

                        key = tuple(sorted([item["Player"].ID for item in lu]))
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
                        st.error(
                            f"Unable to build required lineup #{built+1} for team {team} "
                            f"after {MAX_ATTEMPTS_PER_LINEUP} attempts. Constraints may be impossible."
                        )
                        st.stop()

                    progress.progress(built / max(target, 1))

                    elapsed = time.time() - team_start
                    if built > 0:
                        rate = elapsed / built
                        remaining_time = (target - built) * rate
                        st.caption(f"{team}: ETA ~{remaining_time:.1f} seconds remaining")

                st.success(f"Finished: built {built}/{target} lineups for {team}!")

            if not lineups:
                st.error("Failed to generate any lineups. Relax constraints and try again.")
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
