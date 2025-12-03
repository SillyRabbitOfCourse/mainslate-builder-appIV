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
MIN_SALARY = 49000
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

    # Normalize types
    if "Salary" in df.columns:
        df["Salary"] = df["Salary"].astype(int)
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str)
    if "Position" in df.columns:
        df["Position"] = df["Position"].astype(str)
    if "TeamAbbrev" in df.columns:
        df["TeamAbbrev"] = df["TeamAbbrev"].astype(str)

    return df

# ================================================================
#                     OPPONENT EXTRACTION
# ================================================================

def extract_opponents(df: pd.DataFrame) -> Dict[str, str]:
    """
    Parse 'Game Info' and build map: TEAM -> OPPONENT_TEAM
    Like: IND@JAX -> IND<->JAX
    """
    matchup_map: Dict[str, str] = {}
    if "Game Info" not in df.columns:
        return matchup_map

    for _, row in df.iterrows():
        info = str(row["Game Info"])
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

    if "TeamAbbrev" not in filtered_df.columns or "Name" not in filtered_df.columns:
        return filtered_df

    for team, mode in team_modes.items():
        team_rows = filtered_df["TeamAbbrev"] == team

        if mode == "none":
            continue

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

    if "TeamAbbrev" not in df.columns or "Position" not in df.columns:
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
    if "Position" not in df.columns:
        return groups

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
):
    """
    Select exact players specified in the mini-stack UI.
    Returns a list of 2 pd.Series or None.
    """
    if "Name" not in df.columns or "TeamAbbrev" not in df.columns:
        return None

    if rule["type"] == "same_team":
        team = rule["team"]
        p1_name, p2_name = rule["players"]

        p1 = df[(df["Name"] == p1_name) & (df["TeamAbbrev"] == team)]
        p2 = df[(df["Name"] == p2_name) & (df["TeamAbbrev"] == team)]
        if p1.empty or p2.empty:
            return None

        p1, p2 = p1.iloc[0], p2.iloc[0]
        if getattr(p1, "ID", None) in used_ids or getattr(p2, "ID", None) in used_ids:
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
        if getattr(p1, "ID", None) in used_ids or getattr(p2, "ID", None) in used_ids:
            return None

        return [p1, p2]

    return None

# ================================================================
#               RUN-BACK SELECTION (MODE B: WEIGHTED)
# ================================================================

def select_runbacks_for_stack(team: str, df: pd.DataFrame):
    """
    Mode B:
      - STACK_RUNBACKS[team] stores weights per player (0–1 floats)
      - STACK_RUNBACK_MIN_MAX[team] gives (min, max)
      - We pick K in [min, max] (capped by available players)
      - Then sample K players WITHOUT replacement, weighted by these weights.
    """
    opp = STACK_RUNBACK_TEAMS.get(team, "")
    min_req, max_req = STACK_RUNBACK_MIN_MAX.get(team, (0, 0))

    if not opp:
        return [] if min_req == 0 else None

    pool = get_runback_pool(df, opp)
    if pool.empty:
        return [] if min_req == 0 else None

    weight_map = STACK_RUNBACKS.get(team, {})
    if not weight_map:
        return [] if min_req == 0 else None

    pool = pool[pool["Name"].isin(weight_map.keys())].reset_index(drop=True)
    if pool.empty:
        return [] if min_req == 0 else None

    available = len(pool)
    if available < min_req:
        return None

    max_req = min(max_req, available)
    if min_req > max_req:
        return None

    k = random.randint(min_req, max_req)

    names = pool["Name"].tolist()
    weights = [max(weight_map.get(n, 0.0), 0.0) for n in names]

    if sum(weights) <= 0:
        weights = [1.0] * len(names)

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
):
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

    if "Name" not in df.columns or "TeamAbbrev" not in df.columns or "Position" not in df.columns:
        return None

    # ------------------ REQUIRED PLAYERS ------------------
    required_list = STACK_REQUIRED.get(primary_team, [])
    stack_players = []

    for name in required_list:
        row = df[(df["Name"] == name) & (df["TeamAbbrev"] == primary_team)]
        if row.empty:
            return None
        stack_players.append(row.iloc[0])

    # ------------------ OPTIONAL SPRINKLES ------------------
    sprinkle_names = sample_optional_players(primary_team)
    for name in sprinkle_names:
        row = df[(df["Name"] == name) & (df["TeamAbbrev"] == primary_team)]
        if not row.empty:
            stack_players.append(row.iloc[0])

    # ------------------ RUNBACKS (MODE B) ------------------
    runbacks = select_runbacks_for_stack(primary_team, df)
    if runbacks is None:
        return None
    stack_players.extend(runbacks)

    # ------------------ DST SPRINKLE ------------------
    if STACK_INCLUDE_DST.get(primary_team, False):
        pct = STACK_DST_PERCENT.get(primary_team, 0.0)
        if random.random() < pct:
            dst = df[
                (df["Position"] == "DST") &
                (df["TeamAbbrev"] == primary_team)
            ]
            if not dst.empty:
                stack_players.append(dst.iloc[0])

    used_ids = {getattr(p, "ID", None) for p in stack_players if hasattr(p, "ID")}

    # ------------------ MINI-STACK PLAYERS ------------------
    corr_players = []
    if mini_rule is not None:
        pick = pick_mini_stack_players(
            mini_rule,
            df,
            used_ids,
        )
        if pick is None:
            return None
        for p in pick:
            corr_players.append(p)
            used_ids.add(getattr(p, "ID", None))

    base_players = stack_players + corr_players

    # ------------------ POSITION GROUPS ------------------
    stack_QBs = [p for p in base_players if p.Position == "QB"]
    stack_RBs = [p for p in base_players if p.Position == "RB"]
    stack_WRs = [p for p in base_players if p.Position == "WR"]
    stack_TEs = [p for p in base_players if p.Position == "TE"]
    stack_DSTs = [p for p in base_players if p.Position == "DST"]

    # ------------------ STACK MIN/MAX ENFORCEMENT ------------------
    min_p, max_p = STACK_MIN_MAX.get(primary_team, (2, 5))
    count_primary = sum(1 for p in base_players if p.TeamAbbrev == primary_team)
    if not (min_p <= count_primary <= max_p):
        return None

    # ------------------ MINI EXCLUSIVITY ------------------
    df2 = df.copy()
    if mini_rule is not None and corr_players:
        mini_ids = {getattr(p, "ID", None) for p in corr_players}
        if mini_rule["type"] == "same_team":
            t = mini_rule["team"]
            df2 = df2[
                (df2["TeamAbbrev"] != t) |
                (df2["ID"].isin(mini_ids))
            ]
        elif mini_rule["type"] == "opposing_teams":
            t1, t2 = mini_rule["team1"], mini_rule["team2"]
            df2 = df2[
                ((df2["TeamAbbrev"] != t1) & (df2["TeamAbbrev"] != t2)) |
                (df2["ID"].isin(mini_ids))
            ]

    # ------------------ TEAM ISOLATION FOR FILLERS ------------------
    stack_set = set(STACK_TEAMS)
    runback_set = set(STACK_RUNBACK_TEAMS.values())

    filler_df = df2[
        (~df2["TeamAbbrev"].isin(stack_set)) &
        (~df2["TeamAbbrev"].isin(runback_set))
    ]

    def pool(pos: str):
        subset = filler_df[filler_df["Position"] == pos]
        return subset[~subset.ID.isin(used_ids)].reset_index(drop=True)

    # ------------------ FILL QB ------------------
    if stack_QBs:
        qb = stack_QBs[0]
    else:
        p = pool("QB")
        if p.empty:
            return None
        qb = p.sample(1).iloc[0]
    used_ids.add(qb.ID)

    # ------------------ FILL RB1 & RB2 ------------------
    rbs = stack_RBs.copy()
    while len(rbs) < 2:
        p = pool("RB")
        if p.empty:
            return None
        row = p.sample(1).iloc[0]
        rbs.append(row)
        used_ids.add(row.ID)

    # ------------------ FILL WR1, WR2, WR3 ------------------
    wrs = stack_WRs.copy()
    while len(wrs) < 3:
        p = pool("WR")
        if p.empty:
            return None
        row = p.sample(1).iloc[0]
        wrs.append(row)
        used_ids.add(row.ID)

    # ------------------ FILL TE ------------------
    if stack_TEs:
        te = stack_TEs[0]
    else:
        p = pool("TE")
        if p.empty:
            return None
        te = p.sample(1).iloc[0]
    used_ids.add(te.ID)

    # ------------------ FILL DST ------------------
    if stack_DSTs:
        dst = stack_DSTs[0]
    else:
        p = pool("DST")
        if p.empty:
            return None
        dst = p.sample(1).iloc[0]
    used_ids.add(dst.ID)

    # ------------------ FILL FLEX (NO DST) ------------------
    flex_pool = filler_df[
        (filler_df.Position.isin(FLEX_ELIGIBLE)) &
        (~filler_df.ID.isin(used_ids))
    ]
    if flex_pool.empty:
        return None
    flex = flex_pool.sample(1).iloc[0]

    # ------------------ ASSEMBLE LINEUP ------------------
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
    global STACK_TEAMS, STACK_EXPOSURES, STACK_REQUIRED, STACK_OPTIONAL, STACK_MIN_MAX
    global STACK_RUNBACK_TEAMS, STACK_RUNBACKS, STACK_RUNBACK_MIN_MAX
    global STACK_INCLUDE_DST, STACK_DST_PERCENT
    global MINI_STACKS
    global TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE

    st.title("🏈 Main Slate DFS Lineup Builder — Stacks + Runbacks + Mini-stacks")

    # ------------------- CSV UPLOAD -------------------
    uploaded = st.file_uploader("Upload DKSalaries.csv", type=["csv"])
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
        min_value=1, max_value=200,
        value=NUM_LINEUPS,
    )
    salary_cap = st.sidebar.number_input(
        "Salary cap",
        min_value=10000, max_value=50000,
        value=SALARY_CAP,
    )
    min_salary = st.sidebar.number_input(
        "Minimum salary",
        min_value=0, max_value=salary_cap,
        value=MIN_SALARY,
    )
    seed = st.sidebar.number_input(
        "Random seed (-1 random)",
        value=RANDOM_SEED,
    )

    # ------------------- TABS -------------------
    tab_filters, tab_stacks, tab_runbacks, tab_minis, tab_build = st.tabs(
        ["Global Filters", "Stack Teams", "Run-backs", "Mini-stacks", "Build Lineups"]
    )

    # ================================================================
    #                       GLOBAL FILTERS TAB
    # ================================================================
    with tab_filters:
        st.subheader("Global Team & Player Filters")

        TEAM_FILTER_MODE.clear()
        TEAM_FILTER_KEEP.clear()
        TEAM_FILTER_EXCLUDE.clear()

        for t in all_teams:
            with st.expander(f"Team: {t}", expanded=False):

                mode = st.radio(
                    f"Filter mode for {t}:",
                    ["none", "remove_team", "keep_only", "exclude_only"],
                    key=f"filter_mode_{t}",
                )
                TEAM_FILTER_MODE[t] = mode

                if mode == "keep_only":
                    TEAM_FILTER_KEEP[t] = st.multiselect(
                        "Players to KEEP:",
                        df_raw[df_raw["TeamAbbrev"] == t]["Name"].unique().tolist(),
                        key=f"keep_{t}",
                    )
                elif mode == "exclude_only":
                    TEAM_FILTER_EXCLUDE[t] = st.multiselect(
                        "Players to EXCLUDE:",
                        df_raw[df_raw["TeamAbbrev"] == t]["Name"].unique().tolist(),
                        key=f"exclude_{t}",
                    )
                else:
                    TEAM_FILTER_KEEP[t] = []
                    TEAM_FILTER_EXCLUDE[t] = []

        df_filtered = apply_global_team_filters(
            df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
        )

        st.success(f"Players remaining after filters: {len(df_filtered)}")
        filtered_teams = sorted(df_filtered["TeamAbbrev"].unique().tolist())

    # ================================================================
    #                         STACK TEAMS TAB
    # ================================================================
    with tab_stacks:
        st.subheader("Primary Stack Teams")

        STACK_TEAMS = st.multiselect(
            "Select stack teams:",
            filtered_teams,
            default=[],
        )

        STACK_EXPOSURES.clear()
        STACK_REQUIRED.clear()
        STACK_OPTIONAL.clear()
        STACK_MIN_MAX.clear()

        for team in STACK_TEAMS:
            with st.expander(f"Stack Rules for {team}", expanded=False):

                # Dynamic exposure max
                used = sum(
                    STACK_EXPOSURES.get(t, 0.0)
                    for t in STACK_TEAMS if t != team
                )
                remaining = max(0.0, 1.0 - used)
                current = STACK_EXPOSURES.get(team, 0.0) * 100.0
                if current > remaining * 100.0:
                    current = remaining * 100.0

                exp = st.slider(
                    f"{team} exposure (%)",
                    0.0, remaining * 100.0,
                    current,
                    1.0,
                    key=f"exp_{team}",
                )
                STACK_EXPOSURES[team] = exp / 100.0

                col1, col2 = st.columns(2)
                mn = col1.number_input(
                    f"Min players from {team}",
                    min_value=1, max_value=9,
                    value=2, key=f"min_{team}",
                )
                mx = col2.number_input(
                    f"Max players from {team}",
                    min_value=mn, max_value=9,
                    value=5, key=f"max_{team}",
                )
                STACK_MIN_MAX[team] = (mn, mx)

                # Team players
                team_players = df_filtered[df_filtered["TeamAbbrev"] == team]["Name"].tolist()

                req = st.multiselect(
                    "Required players:",
                    team_players,
                    key=f"req_{team}",
                )
                STACK_REQUIRED[team] = req

                opt = st.multiselect(
                    "Optional sprinkle players:",
                    team_players,
                    key=f"opt_{team}",
                )

                sprinkle_map = {}
                for p in opt:
                    pct = st.slider(
                        f"{p} sprinkle chance (%)",
                        0.0, 100.0, 0.0, 1.0,
                        key=f"sprinkle_pct_{team}_{p}",
                    )
                    sprinkle_map[p] = pct / 100.0

                STACK_OPTIONAL[team] = sprinkle_map

        if STACK_TEAMS:
            st.markdown("### Expected Lineups Per Stack Team")
            for t in STACK_TEAMS:
                pct = STACK_EXPOSURES.get(t, 0.0)
                st.caption(f"{t}: {pct*100:.1f}% → ~{pct * num_lineups:.1f} lineups")





    # ================================================================
    #                           RUN-BACKS TAB
    # ================================================================
    with tab_runbacks:
        st.subheader("Run-backs (Mode B: weighted, min/max aware)")

        STACK_RUNBACK_TEAMS.clear()
        STACK_RUNBACKS.clear()
        STACK_RUNBACK_MIN_MAX.clear()
        STACK_INCLUDE_DST.clear()
        STACK_DST_PERCENT.clear()

        # Helper: compute locked position usage for a given stack team
        def compute_locked_positions_for_stack(team: str) -> Dict[str, int]:
            """
            Count locked position slots for this stack team from:
              - Required players
              - Optional players with 100% sprinkle
            Returns dict like {"QB": x, "RB": y, ...}
            """
            caps = {"QB": 1, "RB": 3, "WR": 4, "TE": 2, "DST": 1}
            locked = {pos: 0 for pos in caps.keys()}

            # Required
            req = STACK_REQUIRED.get(team, [])
            if req:
                r = df_filtered[
                    (df_filtered["TeamAbbrev"] == team) &
                    (df_filtered["Name"].isin(req))
                ]
                for pos in caps.keys():
                    locked[pos] += (r["Position"] == pos).sum()

            # Optional with 100% exposure
            opt_map = STACK_OPTIONAL.get(team, {})
            always = [name for name, pct in opt_map.items() if pct >= 1.0]
            if always:
                r2 = df_filtered[
                    (df_filtered["TeamAbbrev"] == team) &
                    (df_filtered["Name"].isin(always))
                ]
                for pos in caps.keys():
                    locked[pos] += (r2["Position"] == pos).sum()

            return locked

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

                # ------------------ LOCKED POSITION SLOTS ------------------
                locked = compute_locked_positions_for_stack(team)

                # DK caps: RB 3, WR 4, TE 2, DST 1 (QB ignored for runbacks)
                remaining_RB = max(0, 3 - locked.get("RB", 0))
                remaining_WR = max(0, 4 - locked.get("WR", 0))
                remaining_TE = max(0, 2 - locked.get("TE", 0))
                remaining_DST = max(0, 1 - locked.get("DST", 0))

                # ------------------ OPPONENT POOL (NO QB, PRUNE IMPOSSIBLE POS) ------------------
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
                        "No eligible non-QB opponent players remain given the current "
                        "locked positions for this stack team."
                    )
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 0)
                    continue

                # ------------------ PLAYER SELECTION (AUTO-CLEAN) ------------------
                prev_sel = st.session_state.get(f"rbsel_{team}", [])
                prev_sel = [p for p in prev_sel if p in opp_names]

                rb_sel = st.multiselect(
                    f"Run-back players from {opp}:",
                    options=opp_names,
                    default=prev_sel,
                    key=f"rbsel_{team}",
                )

                # ------------------ MIN/MAX SETTINGS ------------------
                max_possible = len(rb_sel)

                # Respect prior values but cap by max_possible
                mn_default = min(st.session_state.get(f"rbmin_{team}", 0), max_possible)
                mx_default = min(st.session_state.get(f"rbmax_{team}", 1), max_possible)

                mn = st.number_input(
                    f"Min run-backs for {team}",
                    min_value=0,
                    max_value=max_possible,
                    value=mn_default,
                    key=f"rbmin_{team}",
                )
                mx = st.number_input(
                    f"Max run-backs for {team}",
                    min_value=mn,
                    max_value=max_possible,
                    value=mx_default,
                    key=f"rbmax_{team}",
                )

                STACK_RUNBACK_MIN_MAX[team] = (mn, mx)

                # ------------------ MODE B WEIGHT SLIDERS ------------------
                rb_map: Dict[str, float] = {}

                for p in rb_sel:
                    slider_key = f"rbpct_{team}_{p}"
                    stored_val = st.session_state.get(slider_key, 0.0)

                    if mn == 1 and mx == 1:
                        # Enforce sum(weights) ≤ 100% when exactly 1 runback required
                        used = 0.0
                        for other in rb_sel:
                            if other == p:
                                continue
                            other_key = f"rbpct_{team}_{other}"
                            # Use latest in rb_map if we've already set it this pass,
                            # otherwise fallback to session_state
                            if other in rb_map:
                                used += rb_map[other]
                            else:
                                used += st.session_state.get(other_key, 0.0) / 100.0

                        remaining = max(0.0, 1.0 - used)
                        current = min(stored_val, remaining * 100.0)

                        pct = st.slider(
                            f"{p} runback weight (%)",
                            0.0,
                            remaining * 100.0,
                            current,
                            1.0,
                            key=slider_key,
                        )
                        rb_map[p] = pct / 100.0
                    else:
                        # Relative weights, no cap sum
                        pct = st.slider(
                            f"{p} runback weight (relative)",
                            0.0, 100.0,
                            stored_val,
                            1.0,
                            key=slider_key,
                        )
                        rb_map[p] = pct / 100.0

                STACK_RUNBACKS[team] = rb_map

                # ------------------ DST SPRINKLE FROM STACK TEAM ------------------
                inc_dst = st.checkbox(
                    f"Allow {team} DST sprinkle?",
                    key=f"dstinc_{team}",
                )
                STACK_INCLUDE_DST[team] = inc_dst

                dst_stored = st.session_state.get(f"dstpct_{team}", 0.0)
                dst_pct = st.slider(
                    f"{team} DST sprinkle chance (%)",
                    0.0, 100.0,
                    dst_stored,
                    1.0,
                    key=f"dstpct_{team}",
                )
                STACK_DST_PERCENT[team] = dst_pct / 100.0



    # ================================================================
    #                         MINI-STACKS TAB
    # ================================================================
    with tab_minis:
        st.subheader("Mini-stacks (secondary correlations)")

        # Make sure mini_rules state exists
        if "mini_rules" not in st.session_state:
            st.session_state["mini_rules"] = []

        mini_rules = st.session_state["mini_rules"]

        # Add new mini-stack buttons
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Add same-team mini-stack"):
                mini_rules.append({
                    "type": "same_team",
                    "team": "",
                    "player1": "",
                    "player2": "",
                    "exposure_pct": 0.0,
                })
        with c2:
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

        # --- Helper: global DK caps ---
        POS_CAPS = {"QB": 1, "RB": 3, "WR": 4, "TE": 2, "DST": 1}

        # --- Helper: count position of a mini pair ---
        def mini_positions(rule):
            """
            Returns dict like {"RB": x, "WR": y, ...}
            """
            counts = {p: 0 for p in POS_CAPS.keys()}
            if rule["type"] == "same_team":
                t = rule["team"]
                p1, p2 = rule["player1"], rule["player2"]
                r1 = df_filtered[(df_filtered["TeamAbbrev"] == t) & (df_filtered["Name"] == p1)]
                r2 = df_filtered[(df_filtered["TeamAbbrev"] == t) & (df_filtered["Name"] == p2)]
                if not r1.empty:
                    counts[r1.iloc[0]["Position"]] += 1
                if not r2.empty:
                    counts[r2.iloc[0]["Position"]] += 1

            elif rule["type"] == "opposing_teams":
                t1, t2 = rule["team1"], rule["team2"]
                p1, p2 = rule["player1"], rule["player2"]

                r1 = df_filtered[(df_filtered["TeamAbbrev"] == t1) & (df_filtered["Name"] == p1)]
                r2 = df_filtered[(df_filtered["TeamAbbrev"] == t2) & (df_filtered["Name"] == p2)]

                if not r1.empty:
                    counts[r1.iloc[0]["Position"]] += 1
                if not r2.empty:
                    counts[r2.iloc[0]["Position"]] += 1

            return counts

        # --- Helper: compute locked positions for any stack team ---
        def locked_slots(team):
            locked = {p: 0 for p in POS_CAPS}
            # Required players
            req = STACK_REQUIRED.get(team, [])
            if req:
                r = df_filtered[
                    (df_filtered["TeamAbbrev"] == team) &
                    (df_filtered["Name"].isin(req))
                ]
                for pos in POS_CAPS:
                    locked[pos] += (r["Position"] == pos).sum()

            # Optional 100%
            opt = STACK_OPTIONAL.get(team, {})
            always = [n for n, pct in opt.items() if pct >= 1.0]
            if always:
                r2 = df_filtered[
                    (df_filtered["TeamAbbrev"] == team) &
                    (df_filtered["Name"].isin(always))
                ]
                for pos in POS_CAPS:
                    locked[pos] += (r2["Position"] == pos).sum()

            return locked

        # -------------------------
        # Mini-stack UI blocks
        # -------------------------
        for i, rule in enumerate(mini_rules):

            with st.expander(f"Mini-stack #{i+1}  ({rule['type']})", expanded=False):

                # ------------------ Exposure slider (global cap = 100%) ------------------
                other_sum = sum(
                    (r.get("exposure_pct", 0.0) / 100.0)
                    for j, r in enumerate(mini_rules)
                    if j != i
                )
                remaining = max(0.0, 1.0 - other_sum)
                current = min(rule.get("exposure_pct", 0.0), remaining * 100.0)

                exp = st.slider(
                    "Exposure (%)",
                    0.0, remaining * 100.0,
                    current,
                    1.0,
                    key=f"mini_exp_{i}",
                )
                rule["exposure_pct"] = exp

                # Delete button
                if st.button("Delete mini-stack", key=f"mini_del_{i}"):
                    remove_idx.append(i)
                    continue

                # ------------------ SAME TEAM MINI SETTINGS ------------------
                if rule["type"] == "same_team":

                    teams_opt = [""] + sorted(df_filtered["TeamAbbrev"].unique().tolist())
                    rule["team"] = st.selectbox(
                        "Team:",
                        teams_opt,
                        index=teams_opt.index(rule["team"]) if rule["team"] in teams_opt else 0,
                        key=f"mini_same_team_{i}",
                    )

                    if rule["team"]:
                        team_players = df_filtered[df_filtered["TeamAbbrev"] == rule["team"]]["Name"].tolist()
                        p_opts = [""] + sorted(team_players)

                        rule["player1"] = st.selectbox(
                            "Player 1:",
                            p_opts,
                            index=p_opts.index(rule["player1"]) if rule["player1"] in p_opts else 0,
                            key=f"mini_same_p1_{i}",
                        )
                        rule["player2"] = st.selectbox(
                            "Player 2:",
                            p_opts,
                            index=p_opts.index(rule["player2"]) if rule["player2"] in p_opts else 0,
                            key=f"mini_same_p2_{i}",
                        )
                    else:
                        rule["player1"] = ""
                        rule["player2"] = ""

                # ------------------ OPPOSING TEAMS MINI SETTINGS ------------------
                elif rule["type"] == "opposing_teams":

                    teams_opt = [""] + sorted(df_filtered["TeamAbbrev"].unique().tolist())
                    rule["team1"] = st.selectbox(
                        "Team 1:",
                        teams_opt,
                        index=teams_opt.index(rule["team1"]) if rule["team1"] in teams_opt else 0,
                        key=f"mini_opp_t1_{i}",
                    )

                    # Auto-opponent (team2)
                    if rule["team1"] in opponent_map:
                        rule["team2"] = opponent_map[rule["team1"]]
                    else:
                        rule["team2"] = ""

                    st.caption(f"Team 2 (opponent): **{rule['team2']}**")

                    # Player pools
                    p1_pool = df_filtered[df_filtered["TeamAbbrev"] == rule["team1"]]["Name"].tolist() if rule["team1"] else []
                    p2_pool = df_filtered[df_filtered["TeamAbbrev"] == rule["team2"]]["Name"].tolist() if rule["team2"] else []

                    p1_opts = [""] + sorted(p1_pool)
                    p2_opts = [""] + sorted(p2_pool)

                    rule["player1"] = st.selectbox(
                        "Player from Team 1:",
                        p1_opts,
                        index=p1_opts.index(rule["player1"]) if rule["player1"] in p1_opts else 0,
                        key=f"mini_opp_p1_{i}",
                    )
                    rule["player2"] = st.selectbox(
                        "Player from Team 2:",
                        p2_opts,
                        index=p2_opts.index(rule["player2"]) if rule["player2"] in p2_opts else 0,
                        key=f"mini_opp_p2_{i}",
                    )

                # ==========================================================
                #       UNIVERSAL FEASIBILITY CHECK (NEW, BIG UPGRADE)
                # ==========================================================
                p1, p2 = rule.get("player1"), rule.get("player2")
                if p1 and p2 and STACK_TEAMS:

                    # Compute positions of the pair
                    mini_pos = mini_positions(rule)

                    feasible = False
                    for t in STACK_TEAMS:
                        locked = locked_slots(t)
                        ok = True

                        # For each position RB/WR/TE/DST ensure locked + mini ≤ cap
                        for pos in POS_CAPS:
                            if locked[pos] + mini_pos[pos] > POS_CAPS[pos]:
                                ok = False
                                break

                        if ok:
                            feasible = True
                            break

                    if not feasible:
                        # Soft error—auto clear invalid mini selection
                        st.info(
                            "This mini-stack cannot fit into ANY stack build "
                            "given positional slot limits. It has been cleared."
                        )
                        rule["player1"] = ""
                        rule["player2"] = ""
                        rule["exposure_pct"] = 0.0
                        st.session_state[f"mini_exp_{i}"] = 0.0

        # Remove deleted minis
        if remove_idx:
            st.session_state["mini_rules"] = [
                r for j, r in enumerate(mini_rules) if j not in remove_idx
            ]

        # Expected usage preview
        if st.session_state["mini_rules"]:
            st.markdown("### Expected Mini-stack Usage:")
            for i, rule in enumerate(st.session_state["mini_rules"]):
                pct = rule.get("exposure_pct", 0.0) / 100.0
                est = NUM_LINEUPS * pct
                st.caption(f"- Mini #{i+1}: {pct*100:.1f}% → ~{est:.1f} lineups")





    # ================================================================
    #                         BUILD LINEUPS TAB
    # ================================================================
    with tab_build:
        st.subheader("Generate Lineups")

        if st.button("🚀 Build Lineups"):

            NUM_LINEUPS = int(num_lineups)
            SALARY_CAP = int(salary_cap)
            MIN_SALARY = int(min_salary)
            RANDOM_SEED = None if seed < 0 else int(seed)


            if RANDOM_SEED is not None:
                random.seed(RANDOM_SEED)

            if not STACK_TEAMS:
                st.info("Please select at least one stack team before building.")
                return

            if sum(STACK_EXPOSURES.get(t, 0.0) for t in STACK_TEAMS) == 0.0:
                st.info("All stack exposures are 0%. Please increase at least one team.")
                return

            # Re-apply filters to be safe
            df_final = apply_global_team_filters(
                df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
            )
            opponent_map_final = extract_opponents(df_final)
            pos_groups = position_split(df_final)

            # Determine how many lineups each stack gets
            stack_counts = {
                team: int(NUM_LINEUPS * STACK_EXPOSURES.get(team, 0.0))
                for team in STACK_TEAMS
            }

            # Build MINI_STACKS config for engine
            MINI_STACKS.clear()
            for rule in st.session_state.get("mini_rules", []):
                pct = rule.get("exposure_pct", 0.0) / 100.0
                if pct <= 0:
                    continue

                if rule["type"] == "same_team":
                    if not (rule.get("team") and rule.get("player1") and rule.get("player2")):
                        continue
                    MINI_STACKS.append({
                        "type": "same_team",
                        "team": rule["team"],
                        "players": [rule["player1"], rule["player2"]],
                        "exposure": pct,
                    })

                elif rule["type"] == "opposing_teams":
                    if not (rule.get("team1") and rule.get("team2") and rule.get("player1") and rule.get("player2")):
                        continue
                    MINI_STACKS.append({
                        "type": "opposing_teams",
                        "team1": rule["team1"],
                        "team2": rule["team2"],
                        "players": [rule["player1"], rule["player2"]],
                        "exposure": pct,
                    })

            # Prepare mini-stack rule state
            mini_state = init_mini_stack_state(NUM_LINEUPS, MINI_STACKS)

            def pick_mini_rule(stack_team: str):
                """
                Return a mini-rule that still has remaining exposure
                and is compatible with this stack team.
                """
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

            # ============================================================
            #                  MAIN GENERATION LOOP
            # ============================================================
            for team in STACK_TEAMS:
                target = stack_counts.get(team, 0)
                if target <= 0:
                    continue

                st.info(f"Building {target} lineups for stack team {team}...")
                progress_bar = st.progress(0)
                start_time = time.time()

                built = 0

                while built < target:
                    attempts = 0
                    success = False

                    while attempts < MAX_ATTEMPTS_PER_LINEUP:
                        attempts += 1

                        # Pick a mini rule if any left
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

                        # Dedup by sorted IDs of players
                        ids = []
                        for entry in lu:
                            player_obj = entry["Player"]
                            pid = getattr(player_obj, "ID", None)
                            if pid is None:
                                # If there's no ID column, fall back to name+team+pos
                                pid = f"{getattr(player_obj, 'Name', '')}-{getattr(player_obj, 'TeamAbbrev', '')}-{getattr(player_obj, 'Position', '')}"
                            ids.append(pid)

                        key = tuple(sorted(ids))
                        if key in used_keys:
                            continue

                        used_keys.add(key)
                        lineups.append(lu)
                        built += 1
                        success = True

                        if m_rule is not None:
                            m_rule["remaining"] -= 1

                        break  # break attempts loop, go build next lineup

                    if not success:
                        st.info(
                            f"Could not find another unique valid lineup for {team} "
                            f"after {MAX_ATTEMPTS_PER_LINEUP} tries. "
                            f"Continuing with {built}/{target} built."
                        )
                        break

                    # Progress + rough ETA
                    progress_bar.progress(built / max(target, 1))
                    elapsed = time.time() - start_time
                    if built > 0:
                        est_total = elapsed / built * target
                        eta = max(0.0, est_total - elapsed)
                        st.caption(
                            f"{team}: {built}/{target} built "
                            f"(~{eta:.1f}s remaining)"
                        )

                st.info(f"Finished {built}/{target} lineups for {team}.")

            # ============================================================
            #                       FINAL OUTPUT
            # ============================================================
            if not lineups:
                st.info("No lineups generated. Relax constraints and try again.")
                return

            st.success(f"Successfully generated {len(lineups)} lineups!")

            # Convert lineups to DataFrame
            def lineups_to_df(lineups_list):
                rows = []
                for i, lu in enumerate(lineups_list, start=1):
                    rec = {"LineupID": i}
                    total_salary = 0

                    for slot in SLOT_ORDER:
                        # find the player with this slot
                        entry = next((x for x in lu if x["Slot"] == slot), None)
                        if entry is None:
                            rec[slot] = ""
                            continue
                        p = entry["Player"]
                        name = getattr(p, "Name", "")
                        pid = getattr(p, "ID", "")
                        sal = getattr(p, "Salary", 0)
                        rec[slot] = f"{name} {pid}"
                        total_salary += sal

                    rec["Total Salary"] = total_salary
                    rows.append(rec)
                return pd.DataFrame(rows)

            df_out = lineups_to_df(lineups)
            st.dataframe(df_out)

            st.download_button(
                "Download Lineups CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="DFS_Lineups.csv",
                mime="text/csv",
            )


# ================================================================
#                         MAIN ENTRY POINT
# ================================================================
if __name__ == "__main__":
    run_app()
