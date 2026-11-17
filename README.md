# Ball knowers — who has the ball?

Our entry for More than a Hack 2027, Phase 1. It labels every frame of Barcelona vs Athletic Club
as **H** (home has the ball), **A** (away has the ball) or **D** (dead ball).

- **Submitted file:** `Ball_knowers.csv`, produced by solution 42. SHA-256 `6a82e233b51acfd3c5ce351c87ff5733b93ad319f2298bbeac2f510fef8c2ec7`.
- **Technical document:** [Ball_knowers.md](Ball_knowers.md).
- **No hand labels, and nothing trained on this match.** Every setting comes from football or from
  seven public matches of another league.

## The approach

**We started by looking at the data, and it had problems.** The automatic tracking loses players,
mostly when they bunch up. It sometimes puts a track on the wrong team, or a goalkeeper in the wrong
half. The ball is not seen on four frames in ten. A method that trusts every number will be wrong
exactly where the game is decided, around the ball.

**So we clean first, and join the two cameras.** The tactical and the broadcast camera each track
the same players and each miss different ones. We synchronise them, repair what can be repaired by
rule, and join them into one set of players per frame. Where the two cameras disagree about a
player, we do not pick a side: that player simply counts for less.

**Then we decide in passes, the way a careful analyst would.**

1. **A first opinion.** A probability model weighs the evidence in each frame: who is nearest the
   ball, how the two teams are moving, whether the players have slowed down, and whether there is a
   reason for play to have stopped. It keeps only the frames it is at least 90% sure of.
2. **A second look at what it was unsure of.** The remaining frames are decided with more
   information than one frame holds: what happened before and after, and football's own logic.
3. **A check against football.** The first opinion is corrected where the game contradicts it. The
   last team to play the ball has it. One touch is not a possession; two plays are. A pass that is not
   intercepted belongs to the passing team. A stoppage needs a cause, and it lasts until the restart.

**Why this and not one trained model.** There is nothing to train on. Hand labels on this match are
forbidden, and the five public minutes hold 17 possessions, too few to fit or rank anything. Rules
can be set from football and checked on other matches, and each one can be read and argued with.

## The method, question by question

All of it is in one file, `experiments/final_chain.py`: twelve small steps, each taking labels in and giving
labels out, each named after the football rule it applies.

| Question | Step | In plain words | Function in `experiments/final_chain.py` |
| --- | --- | --- | --- |
| 1. Trust: what in the data can we believe? | Clean and join | Drop impossible positions, repair tracks given to the wrong team, synchronise the two cameras and join their players into one set per frame; mark players we are unsure of | `context` (tracking version K7, `cleaning.py`) |
| 2. First opinion: what does the evidence say? | Probability model | Weighs nearness to the ball, team movement, slow players and reasons for a stoppage; keeps the frames it is 90% sure of and fills the rest | `context` (`unified_hmm.solution_14_details`) |
| 3. Whose ball? | A change needs a reaction | A short spell of the other team stands only if the teams swap roles or the new team clearly controls the ball | `change_needs_a_reaction` |
| | An unseen change is retimed | A change made while the ball is missing moves to where the teams swap roles | `unseen_change_moves_to_the_role_swap` |
| | Last team to play the ball | Each live frame goes to the team that last played the ball; touches by players we are unsure of do not count; a completed pass stays with the passing team | `last_team_to_play_the_ball` |
| | Two plays | One touch is not a possession | `one_touch_is_not_a_possession`, `two_plays_before_a_whistle` |
| 4. Does it hold up? | A short block needs a reaction | One team for under 4 s inside the other's possession stands only if the teams reacted | `short_block_needs_a_reaction` |
| | The restart check | Whoever restarts did not put the ball out, so the other team's last touch is a possession | `whoever_restarts_did_not_put_it_out` |
| 5. In play: exactly when is play stopped? | Goal kick | One stoppage from the ball crossing the goal line to the kick | `goal_kick_is_one_stoppage` |
| | Placing the ball | "Play" under 5 s between two stoppages is dead if the players stay slow or the ball is only carried | `placing_the_ball_is_not_play` |
| | Whistle with no ball out | Everyone slows to walking pace for 3 s or more with the ball mostly unseen | `whistle_with_no_ball_out` |
| | Shortest stoppage | A stoppage under 3 s is live play | `no_stoppage_under_3_seconds` |
| | Ball clearly out | Either camera shows the ball 2 m or more past a line for a second or more: dead, however short | `ball_clearly_out_is_dead` |

**Why this order.** Each question needs the answer to the one before. Nothing can be decided before we
know which players and ball positions to believe. The first opinion says, roughly, when play is live
and who has the ball. "Whose ball" then settles possession on the frames it calls live, from who
actually played the ball. "Does it hold up" checks that answer against how the teams behave. The
stoppage rules come last, so that nothing after them can undo a stoppage.

We did not start with this order. The steps were added one at a time, each for the next problem we
found. `final_chain.proven()` still runs them in that original order and reproduces our earlier
solution 40 byte for byte, which shows the readable file is the same method. We then tried four
orders of the same twelve steps: they differ on 0.55% to 0.86% of frames and none is clearly better.
Asking "is it in play" before "whose ball" was the one order with a visible cost.

## The possession rules we chose

- Possession starts at the first of two plays. A header straight back to the other team is a contest
  inside their possession.
- A pass in flight belongs to the passer's team. The ball only changes velocity when somebody plays
  it, so a ball that stays straight and never speeds up again was not touched.
- A loose ball stays with the last team to play it, for at most 5 seconds.
- A lone touch with an opponent within reach does not change possession.
- The last touch that puts the ball out is a possession, because the team that restarts cannot be
  the team that put it out.
- Dead ball runs from the ball leaving play, or the whistle, to the restart kick or throw.

## Reproduce the submitted file

You need Python 3.11 and the competition's data bundle. The data is not in this repository and must
not be: get it from the organisers under the terms you accepted.

1. Install the pinned packages:

   ```bash
   pip install -r requirements-ball-knowers.txt
   ```

2. Unpack the data bundle into `data/`, keeping the names it arrives with (`data/README.md` shows
   the layout).

3. Build the file:

   ```bash
   python experiments/submit.py --solution 42 --team Ball_knowers --check-reproducible
   ```

This writes `evaluation/results/Ball_knowers.csv`, validates its format, then rebuilds it from the
raw files with an empty cache and confirms the two are identical. It prints MATCH when the file is
the one we uploaded (the SHA-256 at the top of this page). It takes about a quarter of an hour. Randomness is seeded, and the
time offset between the two cameras is computed from the data, not typed in.

## Outside data: the IDSSE matches

We used one outside dataset, to set a few constants and to test ideas. We labelled nothing ourselves.

- **What it is.** Seven matches of the German Bundesliga and 2. Bundesliga: tracking of every player
  and the ball at 25 frames a second, with the official ball status and ball possession on every
  frame. Data owner: Deutsche Fußball Liga (DFL). Licence: CC BY 4.0.
- **Citation.** Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025). An integrated dataset of
  spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195.
  https://doi.org/10.1038/s41597-025-04505-y
- **Where to get it.** Original files: https://doi.org/10.6084/m9.figshare.28196177. Project page:
  https://github.com/spoho-datascience/idsse-data. Our script reads the mirror at
  https://huggingface.co/datasets/pysport/idsse-data, because the original host blocks scripts.
- **You do not need it to reproduce our file.** The constants taken from it are written in the code,
  each with a comment saying what was measured.

| Constant in the code | Value | Measured on the seven matches |
| --- | --- | --- |
| Weight of team width in the reaction check | 1.31 | How forward speed and width change at a change of possession |
| Shortest stoppage | 3 s | 2.2% of 696 real stoppages are shorter |
| Play between two stoppages that is really the ball being placed | under 5 s, players slow on 90% of it | Lengths of real play between stoppages |
| Sharp slowdown as a reason for a stoppage | mean speed down 2 m/s to 1.5 m/s or less | Followed by a real stoppage 73% of the time |
| Slowdown rule | down 1 m/s to 1.6 m/s or less, for 3 s or more | 92.8% of flagged frames are in a real stoppage |

## What we tried and dropped

Built and measured, then left out of the submitted solution.

| Idea | Why it is out |
| --- | --- |
| Fill the tactical ball's gaps from the broadcast ball | It created extra short possessions. Neither camera's ball is the more accurate |
| A learned model for which touches count | It followed the outside matches' convention, where every touch changes possession |
| One movement path per player | Tracks are pieces of about half a minute with no names; a path stays the same player 75% of a half |
| Place the ball from the players alone | Off by about 6 m: an area, not a position |
| Substitution detection | Not reliable from this tracking, and possession is labelled by team |
| Nearness to the ball as a table of measured probabilities | No gain on held-out matches |

## How good is it, honestly

- **On the five public minutes:** spellF1 0.703 (90% range 0.53 to 0.85), macroF1 0.916. This number
  flatters us. Several steps were written after we looked at cases inside those five minutes. Each
  is a general rule and none refers to a frame of this match, but the cases that prompted them are
  there.
- **On the seven outside matches,** with one camera and no tags: live or dead right on 89% of frames,
  the team right on 87% of live frames, spellF1 between 0.41 and 0.53 depending on how their
  possession label is read (theirs changes on every touch, ours does not).
- **Where it still fails:** stretches with no visible ball, possessions shorter than about 1.5
  seconds, corners and some goal kicks.

## Repository layout

| Path | What |
| --- | --- |
| `experiments/final_chain.py` | The submitted method, start to finish, in one readable file |
| `experiments/pipelines.py` | One function per numbered solution (the history: 01 to 42) |
| `experiments/expected_fingerprints.txt` | SHA-256 of the uploaded file; `submit.py` prints MATCH or MISMATCH against it |
| `experiments/submit.py` | Builds, validates, fingerprints and re-checks the upload file |
| `experiments/*.py` | The steps of the chain, each with a small self-test (`python experiments/<file>.py`) |
| `starter/`, `games.py` | Loading the organisers' files |
| `tag_restart.py` and the other files at the top level | An earlier pipeline of ours; the set-piece step imports it. Some of these files can read the public five minutes for their own reports, and `tag_timing.py` stores that window's frame range for them. Nothing read there is used to produce labels, and the set-piece step is called with the window included, so those frames are treated like every other frame |
| `evaluation/` | The organisers' validator and scorer |
| `requirements-ball-knowers.txt` | Pinned versions |

## Licence

Our code is under the MIT licence in `LICENSE`. The files that came with the competition's starter
repository (`starter/`, `evaluation/`, `games.py`, `DATA.md`, `data/README.md`) are under the organisers'
licence, kept in `LICENSE-organisers`. Neither licence covers the match data.
