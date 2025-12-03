# ================================================================
#                    IMPORTS & GLOBAL CONFIG
# ================================================================

import random
from typing import List, Dict, Tuple
import pandas as pd
import streamlit as st
import re
import time

# ================================================================
#                DEFAULT VALUES (MUST EXIST BEFORE run_app)
# ================================================================

NUM_LINEUPS = 40
SALARY_CAP = 50000
MIN_SALARY = 49000
RANDOM_SEED = 42

SLOT_ORDER = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
FLEX_ELIGIBLE = {"RB", "WR", "TE"}

STACK_TEAMS = []
STACK_EXPOSURES = {}
STACK_REQUIRED = {}
STACK_OPTIONAL = {}
STACK_MIN_MAX = {}

STACK_RUNBACK_TEAMS = {}
STACK_RUNBACKS = {}
STACK_RUNBACK_MIN_MAX = {}

STACK_INCLUDE_DST = {}
STACK_DST_PERCENT = {}

MINI_STACKS = []

TEAM_FILTER_MODE = {}
TEAM_FILTER_KEEP = {}
TEAM_FILTER_EXCLUDE = {}

MAX_ATTEMPTS_PER_LINEUP = 20000
MAX_OVERALL_ATTEMPTS = 40 * 100


# ================================================================
#                          DATA LOADING
# ================================================================

def load_player_pool(source) -> pd.DataFrame:
    """Load DKSalaries.csv and clean column names."""
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]

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
    Parse "Game Info" column and build a mapping:
        TEAM -> OPPONENT_TEAM
    """
    matchup_map = {}

    for _, row in df.iterrows():
        info = row.get("Game Info", "")
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
    """Apply per-team filtering BEFORE stacks/runbacks/minis."""
    filtered_df = df.copy()

    for team, mode in team_modes.items():

        team_rows = filtered_df["TeamAbbrev"] == team

        # No filtering
        if mode == "none":
            continue

        # Remove entire team
        if mode == "remove_team":
            filtered_df = filtered_df[~team_rows]
            continue

        # Keep-only: remove all other players
        if mode == "keep_only":
            keep_list = keep_map.get(team, [])
            filtered_df = filtered_df[
                ~team_rows | filtered_df["Name"].isin(keep_list)
            ]
            continue

        # Exclude-only: remove specified players
        if mode == "exclude_only":
            exclude_list = exclude_map.get(team, [])
            filtered_df = filtered_df[
                ~team_rows | ~filtered_df["Name"].isin(exclude_list)
            ]
            continue

    return filtered_df.reset_index(drop=True)


# ================================================================
#                  POSITION-SPECIFIC DATA GROUPS
# ================================================================

def position_split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group players by position."""
    groups = {}
    for pos, group in df.groupby("Position"):
        groups[pos] = group.reset_index(drop=True)
    return groups


# ================================================================
#                      MINI-STACK SUPPORT UTILS
# ================================================================

def init_mini_stack_state(num_lineups: int, mini_stacks: List[Dict]) -> List[Dict]:
    """
    Convert MINI_STACKS config into a mutable list with a "remaining" counter.
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
    A mini-stack can attach to a stack if:
      - The mini-stack team(s) are NOT the primary stack team
      - Are NOT used as any other stack team
      - Are NOT the opponent runback team for this stack team
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
#                  STACK OPTIONAL SPRINKLE LOGIC
# ================================================================

def sample_optional_players(team: str) -> List[str]:
    """
    Randomly selects optional sprinkle players based on per-player %.
    QBs ARE allowed here.
    """
    chosen = []
    for player, pct in STACK_OPTIONAL.get(team, {}).items():
        if random.random() < pct:
            chosen.append(player)
    return chosen


# ================================================================
#                         RUNBACK SAMPLING
# ================================================================

def get_runback_pool(df: pd.DataFrame, opponent_team: str) -> pd.DataFrame:
    """
    Returns opponent players eligible for runbacks.
    QBs are EXCLUDED.
    """
    if not opponent_team:
        return pd.DataFrame()

    return df[
        (df["TeamAbbrev"] == opponent_team) &
        (df["Position"] != "QB")
    ].reset_index(drop=True)


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

    # how many runbacks in this lineup
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
#                     DST SPRINKLE LOGIC
# ================================================================

def maybe_add_dst_to_stack(team: str, df: pd.DataFrame):
    """
    Add DST from the stack team with a given % chance.
    """
    if not STACK_INCLUDE_DST.get(team, False):
        return None

    pct = STACK_DST_PERCENT.get(team, 0.0)
    if random.random() >= pct:
        return None

    dst = df[(df["Position"] == "DST") & (df["TeamAbbrev"] == team)]
    if dst.empty:
        return None

    return dst.iloc[0]


# ================================================================
#               MINI-STACK PLAYER PICKING (NO QBs)
# ================================================================

def pick_mini_stack_players(rule: Dict, df: pd.DataFrame, used_ids: set) -> List[pd.Series] | None:
    """
    Select exactly 2 players for a mini-stack rule:
      - NO QBs
      - Must be the specific players chosen in UI
      - Cannot duplicate players already used
    """
    players = []

    if rule["type"] == "same_team":
        t = rule["team"]
        p1_name, p2_name = rule["players"]

        r1 = df[(df["TeamAbbrev"] == t) & (df["Name"] == p1_name) & (df["Position"] != "QB")]
        r2 = df[(df["TeamAbbrev"] == t) & (df["Name"] == p2_name) & (df["Position"] != "QB")]

        if r1.empty or r2.empty:
            return None
        if r1.iloc[0].ID in used_ids or r2.iloc[0].ID in used_ids:
            return None

        return [r1.iloc[0], r2.iloc[0]]

    else:  # opposing teams
        t1, t2 = rule["team1"], rule["team2"]
        p1_name, p2_name = rule["players"]

        r1 = df[(df["TeamAbbrev"] == t1) & (df["Name"] == p1_name) & (df["Position"] != "QB")]
        r2 = df[(df["TeamAbbrev"] == t2) & (df["Name"] == p2_name) & (df["Position"] != "QB")]

        if r1.empty or r2.empty:
            return None
        if r1.iloc[0].ID in used_ids or r2.iloc[0].ID in used_ids:
            return None

        return [r1.iloc[0], r2.iloc[0]]

    return None


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
      - Mini team exclusivity (no extra teammates from those mini teams)
      - Team isolation rules for filler
      - Salary cap checks
      - FLEX must be RB/WR/TE (no DST)
    """

    # ------------------ REQUIRED STACK PLAYERS ------------------
    required_list = STACK_REQUIRED.get(primary_team, [])
    stack_players: List[pd.Series] = []

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
    dst_row = maybe_add_dst_to_stack(primary_team, df)
    if dst_row is not None:
        stack_players.append(dst_row)

    # Track used IDs so we don't duplicate
    used_ids = {p.ID for p in stack_players if hasattr(p, "ID")}

    # ------------------ MINI-STACK PLAYERS ------------------
    corr_players: List[pd.Series] = []
    if mini_rule is not None:
        pick = pick_mini_stack_players(mini_rule, df, used_ids)
        if pick is None:
            return None
        for p in pick:
            corr_players.append(p)
            used_ids.add(p.ID)

    # Combine stack + mini prior to position fill
    base_players = stack_players + corr_players

    # ------------------ STACK MIN/MAX ENFORCEMENT ------------------
    min_p, max_p = STACK_MIN_MAX.get(primary_team, (2, 5))
    count_primary = sum(1 for p in base_players if p.TeamAbbrev == primary_team)
    if not (min_p <= count_primary <= max_p):
        return None

    # ------------------ MINI-STACK TEAM EXCLUSIVITY (PER-LINEUP) ------------------
    df2 = df.copy()
    if mini_rule is not None and corr_players:
        mini_ids = {p.ID for p in corr_players}
        if mini_rule["type"] == "same_team":
            team = mini_rule["team"]
            df2 = df2[
                (df2["TeamAbbrev"] != team) |
                (df2["ID"].isin(mini_ids))
            ]
        elif mini_rule["type"] == "opposing_teams":
            t1 = mini_rule["team1"]
            t2 = mini_rule["team2"]
            df2 = df2[
                ((df2["TeamAbbrev"] != t1) & (df2["TeamAbbrev"] != t2)) |
                (df2["ID"].isin(mini_ids))
            ]

    # ------------------ POSITION GROUPS FROM CURRENT BASE ------------------
    stack_QBs = [p for p in base_players if p.Position == "QB"]
    stack_RBs = [p for p in base_players if p.Position == "RB"]
    stack_WRs = [p for p in base_players if p.Position == "WR"]
    stack_TEs = [p for p in base_players if p.Position == "TE"]
    stack_DSTs = [p for p in base_players if p.Position == "DST"]

    # ------------------ TEAM ISOLATION FOR FILLER ------------------
    stack_set = set(STACK_TEAMS)
    runback_set = set(STACK_RUNBACK_TEAMS.values())

    filler_df = df2[
        (~df2["TeamAbbrev"].isin(stack_set)) &
        (~df2["TeamAbbrev"].isin(runback_set))
    ]

    # Helper to get a pool for a position excluding already-used IDs
    def pool(pos: str):
        subset = filler_df[filler_df["Position"] == pos]
        return subset[~subset.ID.isin(used_ids)].reset_index(drop=True)

    # ------------------ FILL QB ------------------
    if stack_QBs:
        qb = stack_QBs[0]
    else:
        qb_pool = pool("QB")
        if qb_pool.empty:
            return None
        qb = qb_pool.sample(1).iloc[0]
    used_ids.add(qb.ID)

    # ------------------ FILL RB1 & RB2 ------------------
    rbs = stack_RBs.copy()
    while len(rbs) < 2:
        rb_pool = pool("RB")
        if rb_pool.empty:
            return None
        row = rb_pool.sample(1).iloc[0]
        rbs.append(row)
        used_ids.add(row.ID)

    # ------------------ FILL WR1, WR2, WR3 ------------------
    wrs = stack_WRs.copy()
    while len(wrs) < 3:
        wr_pool = pool("WR")
        if wr_pool.empty:
            return None
        row = wr_pool.sample(1).iloc[0]
        wrs.append(row)
        used_ids.add(row.ID)

    # ------------------ FILL TE ------------------
    if stack_TEs:
        te = stack_TEs[0]
    else:
        te_pool = pool("TE")
        if te_pool.empty:
            return None
        te = te_pool.sample(1).iloc[0]
    used_ids.add(te.ID)

    # ------------------ FILL DST ------------------
    if stack_DSTs:
        dst = stack_DSTs[0]
    else:
        dst_pool = pool("DST")
        if dst_pool.empty:
            return None
        dst = dst_pool.sample(1).iloc[0]
    used_ids.add(dst.ID)

    # ------------------ FILL FLEX (NO DST, ONLY RB/WR/TE) ------------------
    flex_pool = filler_df[
        (filler_df["Position"].isin(FLEX_ELIGIBLE)) &
        (~filler_df.ID.isin(used_ids))
    ]
    if flex_pool.empty:
        return None
    flex = flex_pool.sample(1).iloc[0]

    # ------------------ ASSEMBLE FINAL LINEUP ------------------
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

    # ------------------ SALARY VALIDATION ------------------
    total_salary = sum(entry["Player"].Salary for entry in lineup)
    if not (MIN_SALARY <= total_salary <= SALARY_CAP):
        return None

    return lineup


# ================================================================
#                  GLOBAL POSITION ANALYZER (NEW)
# ================================================================

def analyze_global_positions(df, stack_required, stack_optional):
    """
    Computes global slot usage + availability across the entire build.
    This becomes the single source of truth for all tabs.
    """
    POS_CAPS = {"QB": 1, "RB": 3, "WR": 4, "TE": 2, "DST": 1}
    used = {pos: 0 for pos in POS_CAPS}
    available = {pos: 0 for pos in POS_CAPS}

    # Count available players in the filtered pool
    for pos in POS_CAPS:
        available[pos] = len(df[df["Position"] == pos])

    # Count required players across ALL stacks
    for team in stack_required:
        for p in stack_required[team]:
            row = df[(df["Name"] == p)]
            if not row.empty:
                pos = row.iloc[0]["Position"]
                used[pos] += 1

    # Count optional 100% sprinkles
    for team in stack_optional:
        for p, pct in stack_optional[team].items():
            if pct >= 1.0:
                row = df[(df["Name"] == p)]
                if not row.empty:
                    pos = row.iloc[0]["Position"]
                    used[pos] += 1

    # Compute remaining capacity
    remain = {pos: POS_CAPS[pos] - used[pos] for pos in POS_CAPS}

    return {
        "caps": POS_CAPS,
        "used": used,
        "remain": remain,
        "available": available
    }




# ================================================================
#                        STREAMLIT UI
# ================================================================

def run_app():
    # All global declarations MUST be at top of function
    global NUM_LINEUPS, SALARY_CAP, MIN_SALARY, RANDOM_SEED
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

        # Clear global filter maps (safe because run_app re-runs every session)
        TEAM_FILTER_MODE.clear()
        TEAM_FILTER_KEEP.clear()
        TEAM_FILTER_EXCLUDE.clear()

        st.caption("Filters apply BEFORE stack selection, runbacks, and minis.")

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

        # Apply filters to shrink player pool
        df_filtered = apply_global_team_filters(
            df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
        )

        st.success(f"Players remaining after filters: {len(df_filtered)}")

        # Update filtered teams
        filtered_teams = sorted(df_filtered["TeamAbbrev"].unique().tolist())

        # -------------------- GLOBAL IMPOSSIBILITY CHECK --------------------
        def global_impossible(df, stack_req, stack_opt):
            pos_data = analyze_global_positions(df, stack_req, stack_opt)

            # If any essential position has ZERO available players → impossible
            essentials = ["QB", "RB", "WR", "TE", "DST"]
            for pos in essentials:
                if pos_data["available"][pos] == 0:
                    return True

            # If FLEX (RB, WR, TE) has no available options
            flex_avail = (
                pos_data["available"]["RB"] +
                pos_data["available"]["WR"] +
                pos_data["available"]["TE"]
            )
            if flex_avail == 0:
                return True

            # If used slots exceed caps anywhere
            for pos in pos_data["caps"]:
                if pos_data["used"][pos] > pos_data["caps"][pos]:
                    return True

            return False
        # -------------------- GLOBAL CONSISTENCY LOCK --------------------
        def enforce_global_consistency(df):
            """
            This is called inside each tab to apply global position logic:
            - Hides impossible positions
            - Prevents contradictory UI states
            - Ensures smooth cross-tab validation
            """
            pos_data = analyze_global_positions(
                df,
                STACK_REQUIRED,
                STACK_OPTIONAL
            )

            return pos_data


    # ================================================================
    #                         STACK TEAMS TAB
    # ================================================================
        # ================================================================
    #                         STACK TEAMS TAB
    # ================================================================
        # ================================================================
    #                         STACK TEAMS TAB
    # ================================================================
        # ================================================================
    #                         STACK TEAMS TAB
    # ================================================================
    with tab_stacks:
        st.subheader("Primary Stack Teams")

        # Choose stack teams AFTER filtering
        STACK_TEAMS = st.multiselect(
            "Select stack teams:",
            filtered_teams,
            default=[],
        )

        # Reset containers
        STACK_EXPOSURES.clear()
        STACK_REQUIRED.clear()
        STACK_OPTIONAL.clear()
        STACK_MIN_MAX.clear()

        # ----------------------------------------------
        # If no stack teams selected, stop here
        # ----------------------------------------------
        if not STACK_TEAMS:
            st.info("Select at least one team to configure stack rules.")
            return

        # ----------------------------------------------
        # Loop through each selected stack team
        # ----------------------------------------------
        for team in STACK_TEAMS:

            with st.expander(f"Stack Rules for {team}", expanded=False):

                # =================================================================
                # 1. GLOBAL POSITION CONSISTENCY OBJECT
                # =================================================================
                pos_data = enforce_global_consistency(df_filtered)

                # =================================================================
                # 2. EXPOSURE SLIDER (auto-capped at global 100%)
                # =================================================================
                already_used = sum(
                    STACK_EXPOSURES.get(t, 0.0)
                    for t in STACK_TEAMS if t != team
                )
                remaining = max(0.0, 1.0 - already_used)
                cur = min(STACK_EXPOSURES.get(team, 0.0) * 100.0, remaining * 100.0)

                exp = st.slider(
                    f"{team} stack exposure (%)",
                    min_value=0.0,
                    max_value=remaining * 100.0,
                    value=cur,
                    step=1.0,
                    key=f"exp_{team}",
                )
                STACK_EXPOSURES[team] = exp / 100.0

                # =================================================================
                # 3. MIN / MAX PLAYERS FROM THIS TEAM
                # =================================================================
                c1, c2 = st.columns(2)
                mn = c1.number_input(
                    f"Min players from {team}",
                    min_value=1, max_value=9,
                    value=2,
                    key=f"min_{team}",
                )
                mx = c2.number_input(
                    f"Max players from {team}",
                    min_value=mn,
                    max_value=9,
                    value=5,
                    key=f"max_{team}",
                )
                STACK_MIN_MAX[team] = (mn, mx)

                # =================================================================
                # 4. TEAM PLAYER LIST (FILTERED BY GLOBAL POSITION LIMITS)
                # =================================================================
                df_team = df_filtered[df_filtered["TeamAbbrev"] == team]

                team_players = []
                for _, r in df_team.iterrows():
                    pos = r["Position"]
                    # Hide players whose position has ZERO slot availability globally
                    if pos_data["remain"][pos] > 0:
                        team_players.append(r["Name"])

                # =================================================================
                # 5. REQUIRED PLAYERS (Mutually exclusive with Optional)
                # =================================================================
                prev_required = [x for x in st.session_state.get(f"req_{team}", []) if x in team_players]

                req = st.multiselect(
                    f"Required players for {team}:",
                    team_players,
                    default=prev_required,
                    key=f"req_{team}",
                )
                STACK_REQUIRED[team] = req

                # =================================================================
                # 6. OPTIONAL SPRINKLE PLAYERS (exclude required)
                # =================================================================
                optional_candidates = [
                    p for p in team_players if p not in req
                ]

                prev_optional = [x for x in st.session_state.get(f"opt_{team}", [])
                                 if x in optional_candidates]

                opt = st.multiselect(
                    f"Optional sprinkle players for {team}:",
                    optional_candidates,
                    default=prev_optional,
                    key=f"opt_{team}",
                )

                sprinkle_map = {}
                for p in opt:
                    stored_val = st.session_state.get(f"sprinkle_pct_{team}_{p}", 0.0)

                    pct = st.slider(
                        f"{p} sprinkle chance (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=stored_val,
                        step=1.0,
                        key=f"sprinkle_pct_{team}_{p}",
                    )
                    sprinkle_map[p] = pct / 100.0

                STACK_OPTIONAL[team] = sprinkle_map

                # =================================================================
                # 7. QB RESTRICTION (only one QB allowed)
                # =================================================================
                qbs_required = [
                    p for p in req
                    if df_team[df_team["Name"] == p].iloc[0]["Position"] == "QB"
                ]
                qbs_optional = [
                    p for p in opt
                    if df_team[df_team["Name"] == p].iloc[0]["Position"] == "QB"
                ]

                # If more than one QB selected across required + optional → auto-fix
                if len(qbs_required) + len(qbs_optional) > 1:
                    st.warning("Only 1 QB allowed per stack team. Adjusting automatically.")
                    # Required QB takes priority
                    if qbs_required:
                        qb_to_keep = qbs_required[0]
                    else:
                        qb_to_keep = qbs_optional[0]

                    # Remove all other QBs
                    for qb in qbs_optional:
                        if qb != qb_to_keep:
                            sprinkle_map.pop(qb, None)
                            st.session_state[f"sprinkle_pct_{team}_{qb}"] = 0.0

                    STACK_OPTIONAL[team] = sprinkle_map

        # ----------------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------------
        st.markdown("### Expected Lineups per Stack Team:")
        for t in STACK_TEAMS:
            pct = STACK_EXPOSURES.get(t, 0.0)
            st.caption(f"{t}: {pct*100:.1f}% → approx {pct * num_lineups:.1f} lineups")



    # ================================================================
    #                           RUN-BACKS TAB
    # ================================================================
        # ================================================================
    #                           RUN-BACKS TAB
    # ================================================================
        # ================================================================
    #                           RUN-BACKS TAB
    # ================================================================
        # ================================================================
    #                           RUN-BACKS TAB
    # ================================================================
        # ================================================================
    #                           RUN-BACKS TAB
    # ================================================================
    with tab_runbacks:
        st.subheader("Run-backs")

        # Reset structures for this tab
        STACK_RUNBACK_TEAMS.clear()
        STACK_RUNBACKS.clear()
        STACK_RUNBACK_MIN_MAX.clear()
        STACK_INCLUDE_DST.clear()
        STACK_DST_PERCENT.clear()

        # ============================================================
        # Helper: count locked positions for a stack team
        # ============================================================
        POS_CAPS = {"QB": 1, "RB": 3, "WR": 4, "TE": 2, "DST": 1}

        def locked_slots(team):
            """
            Required + optional(100%) players for a stack team.
            """
            locked = {p: 0 for p in POS_CAPS}

            # Required
            req_names = STACK_REQUIRED.get(team, [])
            df_req = df_filtered[
                (df_filtered["TeamAbbrev"] == team) &
                (df_filtered["Name"].isin(req_names))
            ]
            for pos in POS_CAPS:
                locked[pos] += (df_req["Position"] == pos).sum()

            # Optional 100% exposure
            opt_map = STACK_OPTIONAL.get(team, {})
            always = [n for n, pct in opt_map.items() if pct >= 1.0]
            df_always = df_filtered[
                (df_filtered["TeamAbbrev"] == team) &
                (df_filtered["Name"].isin(always))
            ]
            for pos in POS_CAPS:
                locked[pos] += (df_always["Position"] == pos).sum()

            return locked

        # ============================================================
        # PROCESS EACH STACK TEAM
        # ============================================================
        for team in STACK_TEAMS:

            opp = opponent_map.get(team, "")
            with st.expander(f"Run-backs for {team}", expanded=False):

                # -------------------------------------------------------
                # No valid opponent
                # -------------------------------------------------------
                if not opp or opp not in filtered_teams:
                    st.info(f"{team} has no valid opponent. Run-backs disabled.")
                    STACK_RUNBACK_TEAMS[team] = ""
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 0)
                    continue

                STACK_RUNBACK_TEAMS[team] = opp
                st.write(f"**Opponent:** {opp}")

                # -------------------------------------------------------
                # Locked slots for the stack team
                # -------------------------------------------------------
                locked = locked_slots(team)

                # -------------------------------------------------------
                # GLOBAL REMAINING POSITION DATA
                # -------------------------------------------------------
                global_pos = enforce_global_consistency(df_filtered)

                # Show slot usage summary
                st.markdown("**Stack Position Usage (before run-backs):**")
                c1, c2, c3, c4 = st.columns(4)
                c1.caption(f"RB {locked['RB']}/{POS_CAPS['RB']}")
                c2.caption(f"WR {locked['WR']}/{POS_CAPS['WR']}")
                c3.caption(f"TE {locked['TE']}/{POS_CAPS['TE']}")
                c4.caption(f"DST {locked['DST']}/{POS_CAPS['DST']}")

                # -------------------------------------------------------
                # Determine local remaining capacity for runbacks
                # -------------------------------------------------------
                remaining_local = {
                    "RB": max(0, POS_CAPS["RB"] - locked["RB"]),
                    "WR": max(0, POS_CAPS["WR"] - locked["WR"]),
                    "TE": max(0, POS_CAPS["TE"] - locked["TE"]),
                    "DST": max(0, POS_CAPS["DST"] - locked["DST"]),
                }

                # Merge global + local restrictions (strictest wins)
                remaining = {
                    pos: min(remaining_local[pos], global_pos["remain"][pos])
                    for pos in ["RB", "WR", "TE", "DST"]
                }

                # -------------------------------------------------------
                # Build the opponent pool (no QB ever)
                # -------------------------------------------------------
                df_opp = df_filtered[
                    (df_filtered["TeamAbbrev"] == opp) &
                    (df_filtered["Position"] != "QB")
                ].copy()

                # Remove positions that have no remaining capacity
                for pos in ["RB", "WR", "TE", "DST"]:
                    if remaining[pos] <= 0:
                        df_opp = df_opp[df_opp["Position"] != pos]

                opp_names = sorted(df_opp["Name"].unique().tolist())

                if not opp_names:
                    st.info(
                        "No eligible opponent players remain for runbacks "
                        "given locked positions + global constraints."
                    )
                    STACK_RUNBACKS[team] = {}
                    STACK_RUNBACK_MIN_MAX[team] = (0, 0)
                    continue

                # -------------------------------------------------------
                # Player selection (auto-clean)
                # -------------------------------------------------------
                prev_sel = st.session_state.get(f"rbsel_{team}", [])
                prev_sel = [p for p in prev_sel if p in opp_names]

                rb_sel = st.multiselect(
                    f"Eligible run-back players from {opp}:",
                    options=opp_names,
                    default=prev_sel,
                    key=f"rbsel_{team}",
                )

                # -------------------------------------------------------
                # Min/Max runback counts
                # -------------------------------------------------------
                max_possible = len(rb_sel)

                # Defaults from session
                mn_default = min(st.session_state.get(f"rbmin_{team}", 0), max_possible)
                mx_default = min(st.session_state.get(f"rbmax_{team}", 1), max_possible)

                # Min first — never exceeds max_possible
                mn = st.number_input(
                    f"Min run-backs for {team}",
                    min_value=0,
                    max_value=max_possible,
                    value=mn_default,
                    key=f"rbmin_{team}",
                )

                # Max should never be below min
                mx = st.number_input(
                    f"Max run-backs for {team}",
                    min_value=mn,
                    max_value=max_possible,
                    value=max(mn, mx_default),
                    key=f"rbmax_{team}",
                )

                STACK_RUNBACK_MIN_MAX[team] = (mn, mx)

                # -------------------------------------------------------
                # Weight sliders
                # -------------------------------------------------------
                rb_map = {}

                # Case 1: Exactly 1 runback required → weights sum ≤ 100%
                if mn == 1 and mx == 1:

                    used_weight = sum(
                        st.session_state.get(f"rbpct_{team}_{other}", 0.0) / 100.0
                        for other in rb_sel
                    )

                    # Build sliders with sum cap
                    for p in rb_sel:
                        slider_key = f"rbpct_{team}_{p}"
                        stored = st.session_state.get(slider_key, 0.0) / 100.0

                        # Remove own contribution temporarily
                        prev_total = (
                            used_weight - stored
                            if p in rb_sel
                            else used_weight
                        )
                        remaining_weight = max(0.0, 1.0 - prev_total)

                        pct = st.slider(
                            f"{p} run-back weight (%)",
                            min_value=0.0,
                            max_value=remaining_weight * 100.0,
                            value=min(stored * 100.0, remaining_weight * 100.0),
                            step=1.0,
                            key=slider_key,
                        )
                        rb_map[p] = pct / 100.0

                # Case 2: 0 or 2–3 runbacks allowed → relative weights only
                else:
                    for p in rb_sel:
                        slider_key = f"rbpct_{team}_{p}"
                        pct = st.slider(
                            f"{p} run-back weight (relative)",
                            min_value=0.0,
                            max_value=100.0,
                            value=st.session_state.get(slider_key, 0.0),
                            step=1.0,
                            key=slider_key,
                        )
                        rb_map[p] = pct / 100.0

                STACK_RUNBACKS[team] = rb_map

                # -------------------------------------------------------
                # DST Sprinkle
                # -------------------------------------------------------
                dst_toggle = st.checkbox(
                    f"Allow {team} DST sprinkle?",
                    value=st.session_state.get(f"dstinc_{team}", False),
                    key=f"dstinc_{team}",
                )
                STACK_INCLUDE_DST[team] = dst_toggle

                dst_pct = st.slider(
                    f"{team} DST sprinkle chance (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get(f"dstpct_{team}", 0.0),
                    step=1.0,
                    key=f"dstpct_{team}",
                )
                STACK_DST_PERCENT[team] = dst_pct / 100.0


    # ================================================================
    #                         MINI-STACKS TAB
    # ================================================================
    with tab_minis:
        st.subheader("Mini-stacks (secondary correlations)")

        # Ensure list exists
        if "mini_rules" not in st.session_state:
            st.session_state["mini_rules"] = []

        mini_rules = st.session_state["mini_rules"]

        # Buttons to add mini rules
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

        # DK caps
        POS_CAPS = {"QB": 1, "RB": 3, "WR": 4, "TE": 2, "DST": 1}

        # Locked slot count for a team
        def locked_slots(team):
            locked = {p: 0 for p in POS_CAPS}

            # Required players
            req = STACK_REQUIRED.get(team, [])
            r = df_filtered[
                (df_filtered["TeamAbbrev"] == team) &
                (df_filtered["Name"].isin(req))
            ]
            for pos in POS_CAPS:
                locked[pos] += (r["Position"] == pos).sum()

            # Optional players with 100% exposure
            opt = STACK_OPTIONAL.get(team, {})
            always = [n for n, pct in opt.items() if pct >= 1.0]
            r2 = df_filtered[
                (df_filtered["TeamAbbrev"] == team) &
                (df_filtered["Name"].isin(always))
            ]
            for pos in POS_CAPS:
                locked[pos] += (r2["Position"] == pos).sum()

            return locked

        # Count pos usage for mini pair
        def mini_positions(rule):
            counts = {p: 0 for p in POS_CAPS}

            if rule["type"] == "same_team":
                team = rule["team"]
                for name in [rule["player1"], rule["player2"]]:
                    if not name:
                        continue
                    row = df_filtered[
                        (df_filtered["TeamAbbrev"] == team) &
                        (df_filtered["Name"] == name)
                    ]
                    if not row.empty:
                        counts[row.iloc[0]["Position"]] += 1

            elif rule["type"] == "opposing_teams":
                t1, p1 = rule["team1"], rule["player1"]
                t2, p2 = rule["team2"], rule["player2"]

                if p1:
                    r1 = df_filtered[
                        (df_filtered["TeamAbbrev"] == t1) &
                        (df_filtered["Name"] == p1)
                    ]
                    if not r1.empty:
                        counts[r1.iloc[0]["Position"]] += 1

                if p2:
                    r2 = df_filtered[
                        (df_filtered["TeamAbbrev"] == t2) &
                        (df_filtered["Name"] == p2)
                    ]
                    if not r2.empty:
                        counts[r2.iloc[0]["Position"]] += 1

            return counts

        # ============================================================
        #                    RENDER EACH MINI-RULE
        # ============================================================
        for i, rule in enumerate(mini_rules):

            with st.expander(f"Mini-stack #{i+1} ({rule['type']})", expanded=False):

                # GLOBAL POSITIONS AVAILABLE
                global_pos = enforce_global_consistency(df_filtered)

                # Exposure slider with auto cap
                other_sum = sum(
                    (r.get("exposure_pct", 0.0) / 100.0)
                    for j, r in enumerate(mini_rules)
                    if j != i
                )
                remaining = max(0.0, 1.0 - other_sum)
                cur_val = min(rule.get("exposure_pct", 0.0), remaining * 100.0)

                exp = st.slider(
                    "Exposure (%)",
                    0.0, remaining * 100.0,
                    cur_val,
                    1.0,
                    key=f"mini_exp_{i}",
                )
                rule["exposure_pct"] = exp

                # Delete button
                if st.button("Delete mini-stack", key=f"mini_del_{i}"):
                    remove_idx.append(i)
                    continue

                # SAME TEAM MINI
                if rule["type"] == "same_team":

                    teams_opt = [""] + sorted(df_filtered["TeamAbbrev"].unique().tolist())
                    rule["team"] = st.selectbox(
                        "Team:",
                        teams_opt,
                        index=teams_opt.index(rule["team"]) if rule["team"] in teams_opt else 0,
                        key=f"mini_same_team_{i}",
                    )

                    if rule["team"]:
                        df_team = df_filtered[df_filtered["TeamAbbrev"] == rule["team"]]

                        team_players = []
                        for _, r in df_team.iterrows():
                            pos = r["Position"]
                            if global_pos["remain"][pos] > 0:
                                team_players.append(r["Name"])

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

                # OPPOSING TEAMS MINI
                elif rule["type"] == "opposing_teams":

                    teams_opt = [""] + sorted(df_filtered["TeamAbbrev"].unique().tolist())
                    rule["team1"] = st.selectbox(
                        "Team 1:",
                        teams_opt,
                        index=teams_opt.index(rule["team1"]) if rule["team1"] in teams_opt else 0,
                        key=f"mini_opp_t1_{i}",
                    )

                    # Auto opponent
                    rule["team2"] = opponent_map.get(rule["team1"], "")

                    st.caption(f"Team 2: **{rule['team2']}**")

                    # Team 1 pool
                    df1 = df_filtered[df_filtered["TeamAbbrev"] == rule["team1"]]
                    p1_pool = []
                    for _, r in df1.iterrows():
                        if global_pos["remain"][r["Position"]] > 0:
                            p1_pool.append(r["Name"])

                    # Team 2 pool
                    df2 = df_filtered[df_filtered["TeamAbbrev"] == rule["team2"]]
                    p2_pool = []
                    for _, r in df2.iterrows():
                        if global_pos["remain"][r["Position"]] > 0:
                            p2_pool.append(r["Name"])

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

                # FINAL FEASIBILITY CHECK
                p1 = rule.get("player1")
                p2 = rule.get("player2")

                if p1 and p2 and STACK_TEAMS:
                    mini_pos = mini_positions(rule)
                    feasible = False

                    for t in STACK_TEAMS:
                        locked = locked_slots(t)
                        ok = True
                        for pos in POS_CAPS:
                            if locked[pos] + mini_pos[pos] > POS_CAPS[pos]:
                                ok = False
                                break
                        if ok:
                            feasible = True
                            break

                    if not feasible:
                        st.info(
                            "Mini-stack exceeds positional caps for all stack teams. "
                            "Resetting selection."
                        )
                        rule["player1"] = ""
                        rule["player2"] = ""
                        rule["exposure_pct"] = 0.0
                        st.session_state[f"mini_exp_{i}"] = 0.0

        # Remove deleted mini rules
        if remove_idx:
            st.session_state["mini_rules"] = [
                r for j, r in enumerate(mini_rules) if j not in remove_idx
            ]

        # Summary
        if st.session_state["mini_rules"]:
            st.markdown("### Expected Mini-stack Usage:")
            for i, rule in enumerate(st.session_state["mini_rules"]):
                pct = rule.get("exposure_pct", 0.0) / 100.0
                est = NUM_LINEUPS * pct
                st.caption(f"- Mini #{i+1}: {pct * 100:.1f}% → ~{est:.1f} lineups")

    #                         BUILD LINEUPS TAB
    # ================================================================
    with tab_build:
        st.subheader("Generate Lineups")

        if st.button("🚀 Build Lineups"):

            # ---------------- UPDATE GLOBAL SETTINGS ----------------
            NUM_LINEUPS = int(num_lineups)
            SALARY_CAP = int(salary_cap)
            MIN_SALARY = int(min_salary)
            RANDOM_SEED = None if seed < 0 else int(seed)

            if RANDOM_SEED is not None:
                random.seed(RANDOM_SEED)

            # ---------------- BASIC PRECHECKS ----------------
            if not STACK_TEAMS:
                st.info("Please select at least one stack team before building.")
                return

            total_exp = sum(STACK_EXPOSURES.get(t, 0.0) for t in STACK_TEAMS)
            if total_exp <= 0:
                st.info("All stack exposures are 0%. Increase at least one stack team’s exposure.")
                return

            # ---------------- FINAL PLAYER POOL ----------------
            df_final = apply_global_team_filters(
                df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
            )
            opponent_map_final = extract_opponents(df_final)
            pos_groups = position_split(df_final)

            # ---------------- GLOBAL IMPOSSIBILITY CHECK ----------------
            if global_impossible(df_final, STACK_REQUIRED, STACK_OPTIONAL):
                st.info(
                    "Current stack / optional / filter configuration is "
                    "mathematically impossible (positions / slots / player pool). "
                    "Relax constraints and try again."
                )
                return

            # ============================================================
            #               MINI-STACK PREPARATION
            # ============================================================
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

            # Convert exposure → remaining counts
            mini_state = init_mini_stack_state(NUM_LINEUPS, MINI_STACKS)

            # Filter out minis that are globally impossible given position caps
            def is_mini_globally_compatible(mr):
                pos_list = []
                for name in mr["players"]:
                    row = df_final[df_final["Name"] == name]
                    if row.empty:
                        return False
                    pos_list.append(row.iloc[0]["Position"])

                pos_data = analyze_global_positions(df_final, STACK_REQUIRED, STACK_OPTIONAL)
                temp_used = pos_data["used"].copy()
                for pos in pos_list:
                    if pos in temp_used:
                        temp_used[pos] += 1

                return all(
                    temp_used[p] <= pos_data["caps"][p]
                    for p in pos_data["caps"]
                )

            mini_state = [mr for mr in mini_state if is_mini_globally_compatible(mr)]

            # ============================================================
            #               STACK LINEUP ALLOCATION
            # ============================================================
            stack_counts = {
                team: int(NUM_LINEUPS * STACK_EXPOSURES.get(team, 0.0))
                for team in STACK_TEAMS
            }

            lineups = []
            used_keys = set()

            import time

            # Helper: pick a mini rule for this stack team
            def pick_mini_rule(stack_team: str):
                for r in mini_state:
                    if r["remaining"] <= 0:
                        continue
                    if mini_rule_applicable_to_team(
                        r, stack_team, STACK_TEAMS, STACK_RUNBACK_TEAMS
                    ):
                        return r
                return None

            # ============================================================
            #                  MAIN GENERATION LOOP
            # ============================================================
            for team in STACK_TEAMS:
                target = stack_counts.get(team, 0)
                if target <= 0:
                    continue

                st.markdown(f"### Building {target} lineups for **{team}**")
                progress_bar = st.progress(0.0)
                start_time = time.time()

                built = 0
                attempts_for_team = 0
                MAX_ATTEMPTS_PER_LINEUP = 5000  # use global cap style, but local here

                while built < target:
                    attempts_for_team += 1
                    if attempts_for_team > MAX_ATTEMPTS_PER_LINEUP:
                        st.info(
                            f"Stopped early: built {built}/{target} lineups for {team}. "
                            "Remaining combinations appear impossible or non-unique "
                            "under current constraints."
                        )
                        break

                    # Pick a compatible mini rule (might be None)
                    m_rule = pick_mini_rule(team)

                    # Try to build a lineup
                    lu = build_stack_lineup(
                        df_final,
                        pos_groups,
                        team,
                        m_rule,
                        opponent_map_final,
                    )

                    if lu is None:
                        continue

                    # Deduplicate by player IDs (or fallback name-team-pos)
                    ids = []
                    for entry in lu:
                        player_obj = entry["Player"]
                        pid = getattr(player_obj, "ID", None)
                        if pid is None:
                            pid = f"{getattr(player_obj, 'Name', '')}-" \
                                  f"{getattr(player_obj, 'TeamAbbrev', '')}-" \
                                  f"{getattr(player_obj, 'Position', '')}"
                        ids.append(pid)

                    key = tuple(sorted(ids))
                    if key in used_keys:
                        continue

                    # Accept lineup
                    used_keys.add(key)
                    lineups.append(lu)
                    built += 1
                    attempts_for_team = 0  # reset attempts on success

                    if m_rule is not None:
                        m_rule["remaining"] -= 1

                    # Progress bar + ETA
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
            #                        FINAL OUTPUT
            # ============================================================
            if not lineups:
                st.info("No lineups generated. Relax constraints and try again.")
                return

            st.success(f"Successfully generated {len(lineups)} lineups!")

            def lineups_to_df(lineups_list):
                rows = []
                for i, lu in enumerate(lineups_list, start=1):
                    rec = {"LineupID": i}
                    total_salary = 0

                    for slot in SLOT_ORDER:
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
                label="Download Lineups CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="DFS_Lineups.csv",
                mime="text/csv",
            )


# ================================================================
#                         MAIN ENTRY POINT
# ================================================================
if __name__ == "__main__":
    run_app()

