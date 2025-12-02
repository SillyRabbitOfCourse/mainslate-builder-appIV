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
SALARY_CAP = 50000            # Corrected salary cap
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

# EXCLUSION RULES based on the new Option A filtering system
TEAM_FILTER_MODE: Dict[str, str] = {}              # "none", "remove_team", "keep_only", "exclude_only"
TEAM_FILTER_KEEP: Dict[str, List[str]] = {}        # For keep-only mode
TEAM_FILTER_EXCLUDE: Dict[str, List[str]] = {}     # For exclude-only mode

MAX_ATTEMPTS_PER_LINEUP = 5000
MAX_OVERALL_ATTEMPTS = 40 * 100



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
    Parse the Game Info column and build a mapping:
    TEAM -> OPPONENT_TEAM
    
    Example:
        IND -> JAX
        JAX -> IND
        SEA -> ATL
        ATL -> SEA
    """
    matchup_map = {}

    for _, row in df.iterrows():
        info = row["Game Info"]
        # Pattern like: "IND@JAX 12/07/2025 01:00PM ET"
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
    This implements the new Option A system.
    """
    filtered_df = df.copy()

    for team, mode in team_modes.items():

        if mode == "none":
            continue

        team_rows = filtered_df["TeamAbbrev"] == team

        # Remove entire team
        if mode == "remove_team":
            filtered_df = filtered_df[~team_rows]
            continue

        # Keep only selected players
        if mode == "keep_only":
            keep_list = keep_map.get(team, [])
            filtered_df = filtered_df[
                ~team_rows | filtered_df["Name"].isin(keep_list)
            ]
            continue

        # Exclude selected players
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
    """
    Randomly select optional sprinkle players for a stack team based on their
    exposure probability. QBs ARE allowed (per your rule) in optional sprinkles.
    """
    chosen = []
    for player_name, pct in STACK_OPTIONAL.get(team, {}).items():
        if random.random() < pct:
            chosen.append(player_name)
    return chosen









# ================================================================
#                         RUNBACK SAMPLING
# ================================================================

def sample_runbacks(team: str) -> List[str]:
    """
    Randomly pick run-back players based on per-player probabilities.
    QBs are automatically filtered out of STACK_RUNBACKS upstream.
    """
    chosen = []
    for player_name, pct in STACK_RUNBACKS.get(team, {}).items():
        if random.random() < pct:
            chosen.append(player_name)
    return chosen












# ================================================================
#                     DST SPRINKLE FOR STACK TEAM
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







# ================================================================
#                DETERMINE IF MINI RULE CAN BE USED
# ================================================================

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

        # Can't use stack or runback teams in minis
        if t1 in stack_set or t2 in stack_set:
            return False
        if t1 in runback_set or t2 in runback_set:
            return False
        if t1 == primary_team or t2 == primary_team:
            return False

        return True

    return False





# ================================================================
#                 MINI-STACK PLAYER PICKING (NO QBs)
# ================================================================

def pick_mini_stack_players(
    rule: Dict,
    df: pd.DataFrame,
    used_ids: set,
    stack_teams: List[str],
    runback_map: Dict[str, str],
) -> List[pd.Series] | None:
    """
    Select 2 players for a mini-stack.
    Cannot use:
    - Stack teams
    - Runback teams
    - QBs anywhere
    - Players already used
    """
    stack_set = set(stack_teams)
    runback_set = set(runback_map.values())

    # Base pool filtered:
    # - No stack teams
    # - No runback teams
    # - No QBs
    # - No already-used IDs
    base = df[
        (~df["TeamAbbrev"].isin(stack_set)) &
        (~df["TeamAbbrev"].isin(runback_set)) &
        (df["Position"] != "QB") &
        (~df["ID"].isin(used_ids))
    ]

    if rule["type"] == "same_team":
        team = rule["team"]
        team_pool = base[base["TeamAbbrev"] == team]
        if team_pool.empty:
            return None

        # Mini stack pairs: [["RB","DST"], ["WR","TE"]] etc
        pos1, pos2 = rule["pairs"][0]     # Only 1 pair per UI design

        p1_pool = team_pool[team_pool["Position"] == pos1]
        if p1_pool.empty:
            return None
        p1 = p1_pool.sample(1).iloc[0]

        remaining = team_pool[team_pool["ID"] != p1["ID"]]
        p2_pool = remaining[remaining["Position"] == pos2]
        if p2_pool.empty:
            return None
        p2 = p2_pool.sample(1).iloc[0]

        return [p1, p2]

    if rule["type"] == "opposing_teams":
        t1 = rule["team1"]
        t2 = rule["team2"]
        pos1, pos2 = rule["pairs"][0]

        pool1 = base[(base["TeamAbbrev"] == t1) & (base["Position"] == pos1)]
        pool2 = base[(base["TeamAbbrev"] == t2) & (base["Position"] == pos2)]

        if pool1.empty or pool2.empty:
            return None

        p1 = pool1.sample(1).iloc[0]
        p2 = pool2.sample(1).iloc[0]

        return [p1, p2]

    return None






# ================================================================
#                           RUN-BACK SYSTEM
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
        (df["Position"] != "QB")  # QBs never allowed as runbacks
    ].reset_index(drop=True)






# ================================================================
#                     RUN-BACK PLAYER PREPARATION
# ================================================================

def prepare_runback_candidates(team: str, df: pd.DataFrame) -> List[pd.Series]:
    """
    Gather ALL possible run-back players for a given stack team,
    filtered by opponent + non-QB rule.
    """
    opp = STACK_RUNBACK_TEAMS.get(team, "")
    if not opp:
        return []

    pool = get_runback_pool(df, opp)
    return list(pool.itertuples(index=False))





# ================================================================
#               RUN-BACK SELECTION WITH MIN/MAX ENFORCEMENT
# ================================================================

def select_runbacks_for_stack(team: str, df: pd.DataFrame) -> List[pd.Series]:
    """
    Select run-back players for a stack team using:
      - Opponent detection
      - Player probabilities
      - Min/max enforcement
    """
    opp = STACK_RUNBACK_TEAMS.get(team, "")
    if not opp:
        # If UI requires min runbacks but no opponent is set → impossible
        min_req, _ = STACK_RUNBACK_MIN_MAX.get(team, (0, 999))
        return [] if min_req == 0 else None

    # Full pool of opponent players (non-QB)
    pool = get_runback_pool(df, opp)
    if pool.empty:
        min_req, _ = STACK_RUNBACK_MIN_MAX.get(team, (0, 999))
        return [] if min_req == 0 else None

    # Probabilistic run-back triggers
    chosen = []
    for player_name, pct in STACK_RUNBACKS.get(team, {}).items():
        if random.random() < pct:
            row = pool[pool["Name"] == player_name]
            if not row.empty:
                chosen.append(row.iloc[0])

    # Enforce minimum
    min_req, max_req = STACK_RUNBACK_MIN_MAX.get(team, (0, 999))
    chosen_ids = set([p.ID for p in chosen])

    if len(chosen) < min_req:
        needed = min_req - len(chosen)

        # available pool EXCLUDING already chosen
        available = pool[~pool["ID"].isin(chosen_ids)]
        if len(available) < needed:
            return None  # cannot satisfy min → lineup attempt fails

        extra = available.sample(n=needed)
        for _, row in extra.iterrows():
            chosen.append(row)

    # Enforce maximum
    if len(chosen) > max_req:
        chosen = random.sample(chosen, max_req)

    return chosen





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
      - Run-backs (correct opponent only)
      - Mini-stacks (if allowed)
      - Team isolation rules
      - Salary cap checks
      - FLEX must be RB/WR/TE (no DST)
    """

    # ---------------------------------------------------
    # PRIMARY STACK REQUIRED PLAYERS
    # ---------------------------------------------------
    required_list = STACK_REQUIRED.get(primary_team, [])
    stack_players: List[pd.Series] = []

    for name in required_list:
        row = df[(df["Name"] == name) & (df["TeamAbbrev"] == primary_team)]
        if row.empty:
            return None
        stack_players.append(row.iloc[0])

    # ---------------------------------------------------
    # OPTIONAL SPRINKLES
    # ---------------------------------------------------
    sprinkle_names = sample_optional_players(primary_team)
    for name in sprinkle_names:
        row = df[(df["Name"] == name) & (df["TeamAbbrev"] == primary_team)]
        if not row.empty:
            stack_players.append(row.iloc[0])

    # ---------------------------------------------------
    # RUN-BACK HANDLING
    # ---------------------------------------------------
    runbacks = select_runbacks_for_stack(primary_team, df)
    if runbacks is None:  # could not satisfy min runbacks
        return None

    for rb in runbacks:
        stack_players.append(rb)

    # ---------------------------------------------------
    # DST SPRINKLE (from primary team only)
    # ---------------------------------------------------
    dst_row = maybe_add_dst_to_stack(primary_team, df)
    if dst_row is not None:
        stack_players.append(dst_row)

    # ---------------------------------------------------
    # PREP FOR ROSTER CONSTRUCTION
    # ---------------------------------------------------
    used_ids = {p.ID for p in stack_players}
    player_objs = list(stack_players)

    # ---------------------------------------------------
    # OPTIONAL MINI-STACK (must NOT use stack or runback teams)
    # ---------------------------------------------------
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

    # Combine everything prior to filling positions
    base_players = player_objs + corr_players

    # ---------------------------------------------------
    # BEGIN POSITION FILLING
    # ---------------------------------------------------

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

    # ---------------------------------------------------
    # TEAM ISOLATION LOGIC
    # ---------------------------------------------------
    stack_set = set(STACK_TEAMS)
    runback_set = set(STACK_RUNBACK_TEAMS.values())

    # filler only from teams that are:
    # - NOT stack teams
    # - NOT run-back teams
    # - NOT primary-team
    filler_df = df[
        (~df["TeamAbbrev"].isin(stack_set)) &
        (~df["TeamAbbrev"].isin(runback_set))
    ]

    # ---------------------------------------------------
    # POOL HELPER
    # ---------------------------------------------------
    def pool(pos: str):
        subset = filler_df[filler_df.Position == pos]
        return subset[~subset.ID.isin(used_ids)].reset_index(drop=True)

    # ---------------------------------------------------
    # FILL QB
    # ---------------------------------------------------
    if stack_QBs:
        qb = stack_QBs[0]
    else:
        p = pool("QB")
        if p.empty:
            return None
        qb = p.sample(1).iloc[0]

    used_ids.add(qb.ID)

    # ---------------------------------------------------
    # FILL RB1 & RB2
    # ---------------------------------------------------
    rbs = stack_RBs.copy()

    while len(rbs) < 2:
        p = pool("RB")
        if p.empty:
            return None
        row = p.sample(1).iloc[0]
        rbs.append(row)
        used_ids.add(row.ID)

    # ---------------------------------------------------
    # FILL WR1, WR2, WR3
    # ---------------------------------------------------
    wrs = stack_WRs.copy()

    while len(wrs) < 3:
        p = pool("WR")
        if p.empty:
            return None
        row = p.sample(1).iloc[0]
        wrs.append(row)
        used_ids.add(row.ID)

    # ---------------------------------------------------
    # FILL TE
    # ---------------------------------------------------
    if stack_TEs:
        te = stack_TEs[0]
    else:
        p = pool("TE")
        if p.empty:
            return None
        te = p.sample(1).iloc[0]

    used_ids.add(te.ID)

    # ---------------------------------------------------
    # FILL DST
    # ---------------------------------------------------
    if stack_DSTs:
        dst = stack_DSTs[0]
    else:
        p = pool("DST")
        if p.empty:
            return None
        dst = p.sample(1).iloc[0]

    used_ids.add(dst.ID)

    # ---------------------------------------------------
    # FILL FLEX  (NO DST ALLOWED)
    # ---------------------------------------------------
    flex_pool = filler_df[
        (filler_df.Position.isin(FLEX_ELIGIBLE)) &
        (~filler_df.ID.isin(used_ids))
    ]
    if flex_pool.empty:
        return None
    flex = flex_pool.sample(1).iloc[0]

    # ---------------------------------------------------
    # ASSEMBLE FINAL LINEUP
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # SALARY VALIDATION
    # ---------------------------------------------------
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
    global MAX_OVERALL_ATTEMPTS

    st.title("🏈 Main Slate DFS Lineup Builder — Stacks + Runbacks + Mini-stacks")

    # ------------------- CSV UPLOAD -------------------
    uploaded = st.file_uploader("Upload a DKSalaries.csv file", type=["csv"])
    if not uploaded:
        st.info("Please upload a **DKSalaries.csv** file to continue.")
        return

    # Load the raw data now
    df_raw = load_player_pool(uploaded)
    all_teams = sorted(df_raw["TeamAbbrev"].unique().tolist())
    all_players = sorted(df_raw["Name"].unique().tolist())

    # Build opponent map from Game Info
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
        Filtering here happens *before* stack/runback/min-stack selection.
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

        # Apply global filters to the DF for the following tabs
        df_filtered = apply_global_team_filters(
            df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
        )
        filtered_players = sorted(df_filtered["Name"].unique().tolist())
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

        if STACK_TEAMS:
            st.markdown("### Configure Stack Rules")

        STACK_EXPOSURES.clear()
        STACK_REQUIRED.clear()
        STACK_OPTIONAL.clear()
        STACK_MIN_MAX.clear()

        for team in STACK_TEAMS:

            with st.expander(f"Stack Team: {team}", expanded=False):

                # ---------------- Exposure % ----------------
                exp = st.slider(
                    f"{team} stack exposure (%)",
                    0.0, 100.0, 0.0, 1.0,
                    key=f"exposure_{team}",
                )
                STACK_EXPOSURES[team] = exp / 100.0

                # ---------------- Min/Max ----------------
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

                # ---------------- Required players (QB allowed) ----------------
                team_players = sorted(
                    df_filtered[df_filtered["TeamAbbrev"] == team]["Name"].unique().tolist()
                )

                required = st.multiselect(
                    f"Required players ({team}):",
                    options=team_players,
                    key=f"required_{team}",
                )
                STACK_REQUIRED[team] = required

                # ---------------- Optional players (QB allowed) ----------------
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




    # ================================================================
    #                           RUN-BACKS
    # ================================================================
    with tab_runbacks:
        st.subheader("Run-back Settings (Automatically Uses Correct Opponent)")

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

                # AUTOMATIC OPPONENT
                STACK_RUNBACK_TEAMS[team] = opp
                st.write(f"**Opponent:** {opp}")

                # Opponent player pool (NO QBs)
                opp_pool = df_filtered[
                    (df_filtered["TeamAbbrev"] == opp) &
                    (df_filtered["Position"] != "QB")
                ]
                opp_names = sorted(opp_pool["Name"].unique().tolist())

                rb_sel = st.multiselect(
                    f"Eligible run-back players from {opp}:",
                    options=opp_names,
                    key=f"rbsel_{team}",
                )

                rb_map = {}
                for p in rb_sel:
                    pct = st.slider(
                        f"{p} run-back chance (%)",
                        0.0, 100.0, 0.0, 1.0,
                        key=f"rbpct_{team}_{p}",
                    )
                    rb_map[p] = pct / 100.0

                STACK_RUNBACKS[team] = rb_map

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
                STACK_RUNBACK_MIN_MAX[team] = (mn, mx)

                # DST toggle
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
                    "pos1": "RB",
                    "pos2": "DST",
                    "exposure_pct": 0.0,
                })
        with col_add2:
            if st.button("➕ Add opposing-team mini-stack"):
                mini_rules.append({
                    "type": "opposing_teams",
                    "team1": "",
                    "team2": "",
                    "pos1": "WR",
                    "pos2": "WR",
                    "exposure_pct": 0.0,
                })

        remove_idx = []

        positions = ["RB","WR","TE","DST"]  # QBs excluded globally for minis

        for i, rule in enumerate(mini_rules):
            with st.expander(f"Mini-Stack #{i+1} ({rule['type']})"):

                # Exposure
                exp = st.slider(
                    "Exposure (%)", 0.0, 100.0, rule["exposure_pct"], 1.0,
                    key=f"mini_exp_{i}"
                )
                rule["exposure_pct"] = exp

                # Delete
                if st.button("Delete", key=f"mini_del_{i}"):
                    remove_idx.append(i)
                    continue

                if rule["type"] == "same_team":
                    rule["team"] = st.selectbox(
                        "Team:",
                        options=[""] + filtered_teams,
                        index=([""] + filtered_teams).index(rule["team"]) if rule["team"] in filtered_teams else 0,
                        key=f"mini_team_{i}",
                    )
                    rule["pos1"] = st.selectbox(
                        "Position 1:", positions,
                        key=f"mini_pos1_{i}",
                    )
                    rule["pos2"] = st.selectbox(
                        "Position 2:", positions,
                        key=f"mini_pos2_{i}",
                    )

                elif rule["type"] == "opposing_teams":

                    t1 = st.selectbox(
                        "Team 1:",
                        options=[""] + filtered_teams,
                        index=([""] + filtered_teams).index(rule["team1"]) if rule["team1"] in filtered_teams else 0,
                        key=f"mini_t1_{i}",
                    )
                    rule["team1"] = t1

                    # Team2 must be opponent
                    t2_options = [""] + ([opponent_map.get(t1)] if t1 in opponent_map else [])
                    rule["team2"] = st.selectbox(
                        "Team 2 (auto opponent):",
                        options=t2_options,
                        index=t2_options.index(rule["team2"]) if rule["team2"] in t2_options else 0,
                        key=f"mini_t2_{i}",
                    )

                    rule["pos1"] = st.selectbox(
                        "Position (Team 1):", positions,
                        key=f"mini_pos1_opp_{i}",
                    )
                    rule["pos2"] = st.selectbox(
                        "Position (Team 2):", positions,
                        key=f"mini_pos2_opp_{i}",
                    )

        # Remove deleted mini-rules
        if remove_idx:
            st.session_state["mini_rules"] = [
                r for idx, r in enumerate(mini_rules) if idx not in remove_idx
            ]






    # ================================================================
    #                         BUILD LINEUPS TAB
    # ================================================================
        # ================================================================
    #                         BUILD LINEUPS TAB
    # ================================================================
    with tab_build:
        st.subheader("Generate Lineups")

        # Summaries
        if STACK_TEAMS:
            st.write("### Configuration Summary:")
            st.write(f"Primary stack teams: **{', '.join(STACK_TEAMS)}**")

        # ---------------- BUILD BUTTON ----------------
        if st.button("🚀 Build Lineups"):
            # -------------------------------------------
            # Set global variables
            # -------------------------------------------
            NUM_LINEUPS = int(num_lineups)
            SALARY_CAP = int(salary_cap)
            MIN_SALARY = int(min_salary)
            RANDOM_SEED = None if seed < 0 else int(seed)

            if RANDOM_SEED is not None:
                random.seed(RANDOM_SEED)

            if not STACK_TEAMS:
                st.error("Please select at least one stack team before building.")
                return

            # Exposure sanity check
            if sum(STACK_EXPOSURES.get(t, 0.0) for t in STACK_TEAMS) == 0.0:
                st.error("All stack exposures are 0%. Please increase at least one team.")
                return

            # -------------------------------------------
            # Final filtered pool
            # -------------------------------------------
            df_final = apply_global_team_filters(
                df_raw, TEAM_FILTER_MODE, TEAM_FILTER_KEEP, TEAM_FILTER_EXCLUDE
            )

            # Final opponent map
            opponent_map_final = extract_opponents(df_final)

            # Position groups
            pos_groups = position_split(df_final)

            # -------------------------------------------
            # Determine stack counts
            # -------------------------------------------
            stack_counts = {
                team: int(NUM_LINEUPS * STACK_EXPOSURES.get(team, 0.0))
                for team in STACK_TEAMS
            }
            st.write("### Planned Lineups Per Stack:")
            st.json(stack_counts)

            # -------------------------------------------
            # Prepare mini-stack rule state
            # -------------------------------------------
            MINI_STACKS = []
            for rule in st.session_state.get("mini_rules", []):
                pct = rule.get("exposure_pct", 0.0) / 100.0
                if pct <= 0:
                    continue

                if rule["type"] == "same_team":
                    if not rule["team"]:
                        continue
                    MINI_STACKS.append({
                        "type": "same_team",
                        "team": rule["team"],
                        "exposure": pct,
                        "pairs": [[rule["pos1"], rule["pos2"]]],
                    })

                elif rule["type"] == "opposing_teams":
                    if not rule["team1"] or not rule["team2"]:
                        continue
                    MINI_STACKS.append({
                        "type": "opposing_teams",
                        "team1": rule["team1"],
                        "team2": rule["team2"],
                        "exposure": pct,
                        "pairs": [[rule["pos1"], rule["pos2"]]],
                    })

            mini_state = init_mini_stack_state(NUM_LINEUPS, MINI_STACKS)

            # Helper for mini-rule selection
            def pick_mini_rule(stack_team):
                for r in mini_state:
                    if r["remaining"] <= 0:
                        continue
                    if mini_rule_applicable_to_team(
                        r, stack_team, STACK_TEAMS, STACK_RUNBACK_TEAMS
                    ):
                        return r
                return None

            # -------------------------------------------
            # GUARANTEED LINEUP GENERATION (NEW)
            # -------------------------------------------
            import time

            lineups = []
            used_keys = set()
            MAX_ATTEMPTS_PER_LINEUP = 20000

            for team in STACK_TEAMS:
                target = stack_counts.get(team, 0)
                built = 0

                st.info(f"Building {target} lineups for team {team}...")

                # Progress bar
                progress = st.progress(0)

                # Start timer
                team_start = time.time()

                # Start building
                while built < target:

                    attempts_for_this_lineup = 0
                    success = False

                    # Attempt until success or cap
                    while attempts_for_this_lineup < MAX_ATTEMPTS_PER_LINEUP:
                        attempts_for_this_lineup += 1

                        # Select mini-rule if any remain
                        m_rule = pick_mini_rule(team)

                        # Attempt lineup
                        lu = build_stack_lineup(
                            df_final,
                            pos_groups,
                            team,
                            m_rule,
                            opponent_map_final,
                        )

                        if lu is None:
                            continue

                        # Dedup
                        key = tuple(sorted([item["Player"].ID for item in lu]))
                        if key in used_keys:
                            continue

                        # SUCCESS
                        used_keys.add(key)
                        lineups.append(lu)
                        built += 1
                        success = True

                        # Reduce mini-rule remaining if used
                        if m_rule is not None and "remaining" in m_rule:
                            m_rule["remaining"] -= 1

                        break  # lineup successfully built

                    # Safety error
                    if not success:
                        st.error(
                            f"Unable to build required lineup #{built+1} for team {team} "
                            f"after {MAX_ATTEMPTS_PER_LINEUP} attempts. Constraints may be impossible."
                        )
                        st.stop()

                    # Update progress bar
                    progress.progress(built / target)

                    # ETA update
                    elapsed = time.time() - team_start
                    if built > 0:
                        rate = elapsed / built
                        remaining = (target - built) * rate
                        st.caption(f"ETA: {remaining:.1f} seconds remaining")

                st.success(f"Finished: built {built}/{target} lineups for {team}!")

            # -------------------------------------------
            # Final checks
            # -------------------------------------------
            if not lineups:
                st.error("Failed to generate any lineups. Relax constraints and try again.")
                return

            st.success(f"Successfully generated {len(lineups)} lineups!")

            # -------------------------------------------
            # Convert to DF
            # -------------------------------------------
            def lineups_to_df(lineups):
                rows = []
                for i, lu in enumerate(lineups, start=1):
                    rec = {"LineupID": i}
                    total = 0
                    for slot in SLOT_ORDER:
                        p = next(item["Player"] for item in lu if item["Slot"] == slot)
                        rec[slot] = p["Name"] + " " + str(p["ID"])
                        total += p["Salary"]
                    rec["Total Salary"] = total
                    rows.append(rec)
                return pd.DataFrame(rows)

            df_out = lineups_to_df(lineups)
            st.dataframe(df_out)

            # -------------------------------------------
            # Download button
            # -------------------------------------------
            st.download_button(
                label="Download Lineups CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="DFS_Lineups.csv",
                mime="text/csv"
            )

# ================================================================
#                           APP ENTRY POINT
# ================================================================

if __name__ == "__main__":
    run_app()
