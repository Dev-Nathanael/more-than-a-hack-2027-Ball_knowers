# Ball knowers — technical document

More than a Hack 2027, Phase 1: every frame of Barcelona vs Athletic Club labelled H (home has the
ball), A (away) or D (dead ball).

**Submitted file:** `Ball_knowers.csv`, produced by solution 42 of the repository.
SHA-256 `6a82e233b51acfd3c5ce351c87ff5733b93ad319f2298bbeac2f510fef8c2ec7`.

## 1. The approach

Five questions asked in order, each answered by a few small steps. The first steps decide what in the
tracking can be believed, a probability model gives a first opinion, and football rules then settle who
has the ball, check that answer, and fix exactly when play is stopped. Nothing is learned from human
labels of this match.

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

**Why a chain and not one trained model.** There are no labels to train on: hand labels on this
match are forbidden, and the public five minutes hold 17 possessions, too few to fit or rank
anything. A chain of explicit rules can be set from football and from other matches, and each step
can be inspected.

**Using two cameras without trusting either blindly.** The two cameras are joined (step 1), and a
touch is ignored when the player who made it is seen by one camera only, or is put on different
teams by the two cameras. Such a player's position or team is unconfirmed, so his touch is evidence
for neither team. The broadcast camera's ball is used for two things only: synchronising the
cameras, and saying that the ball is off the pitch (step 10).

## 2. Assets used, and dropped

| Asset | Used for | Notes |
| --- | --- | --- |
| Tactical tracking | Everything | Players and ball in pitch coordinates |
| Broadcast tracking, players | Joining the cameras; a second opinion on a player's team | Synchronised to the tactical timeline by sliding one ball path over the other; the offset is computed, not typed in |
| Broadcast tracking, ball | Synchronisation, and "the ball is off the pitch" | Tried as a fill for the tactical ball's gaps and dropped: it created extra short possessions. Neither camera's ball is the more accurate; they disagree mostly while the ball is in flight |
| Smart Tagging | Set-piece and goal tags as reasons for a stoppage | Automatic tags, as provided |
| Homography | Not used by the pipeline | The tracking is already in pitch coordinates. Used by our local viewer only |
| Video and audio | Not used by the pipeline | Used by our local viewer. The audio track was never analysed |

Also built and dropped: a learned
model for which touches count; an alternative goalkeeper repair; one movement path per player;
placing the ball from the players alone; substitution detection; nearness to the ball as a table of
measured probabilities.

## 3. Training data

**No labelled data from the provided game was used**, in any amount or for any purpose.

**Outside data: the IDSSE dataset.** Seven matches of the German Bundesliga and 2. Bundesliga,
data owner Deutsche Fußball Liga (DFL), licence CC BY 4.0:
Bassek, M., Rein, R., Weber, H. and Memmert, D. (2025), "An integrated dataset of spatiotemporal
and event data in elite soccer", *Scientific Data* 12, 195, https://doi.org/10.1038/s41597-025-04505-y
(files: https://doi.org/10.6084/m9.figshare.28196177). It contains 25 frames-a-second tracking of all
players and the ball, with a ball status (in play or not) and a ball possession label on every
frame. Those labels are the official DFL match data, not ours; we labelled nothing.

It was used to set a small number of constants, never to fit a model that labels our match:

| Constant | Value | What was measured on the seven matches |
| --- | --- | --- |
| Weight of team width in the reaction check | 1.31 | How the teams' forward speed and width change at a change of possession |
| Shortest stoppage | 3 s | 2.2% of 696 real stoppages are shorter |
| "Play" between two stoppages that is really the ball being placed | under 5 s, players slow on 90% of it | Lengths of real play between stoppages, and how slow players are inside stoppages |
| Whole-team reaction, not the five nearest players | — | The whole team separates a real short block from calm play better (0.80 against 0.77) |
| Sharp collective slowdown as a reason for a stoppage | mean player speed down 2 m/s to 1.5 m/s or less | Such a slowdown is followed by a real stoppage 73% of the time |
| Slowdown rule (step 9) | down 1 m/s to 1.6 m/s or less, for 3 s or more, ball seen on at most half | With the ball hidden as on our match, 92.8% of the frames it flags are in a real stoppage; 15 false stretches in 101 |

What it could not be used for: anything that reads the ball during a stoppage (the dataset parks the
ball on every dead frame), anything using tags or a second camera, and how often possession changes
(its label changes on every single touch; ours does not).

**Pseudo-labels on the provided game.** Two small logistic models (team shape, team movement) are
fitted on frames where the chain itself is confident: the ball is seen, one team clearly controls
it, and it moves with its player. No person chose any of those frames.

**Where the remaining constants come from.** Football and the Laws of the Game (reach of a player,
speed of a kicked ball, where a goal kick is taken), and the match's own consistency (how fast
tracked players can move, how the two cameras agree).

## 4. The possession rules implemented

- **Possession starts at the first of two plays.** One touch is not a possession. A header that goes
  straight back to the other team, or a block, is a contest inside their possession.
- **A pass in flight belongs to the passer's team** until another team plays the ball. The ball only
  changes velocity when somebody plays it: a ball that stays straight and never speeds up again was
  not touched, however close it passed to an opponent.
- **Whose play was it?** A play made within a second of a team's own touch, with that team still
  within reach, is that team's play, even if an opponent stands nearer.
- **A loose ball stays with the last team to play it**, for at most 5 seconds.
- **A challenge:** a lone touch made with an opponent within reach does not change possession.
- **The last touch that puts the ball out is a possession**, even though it is a single touch,
  because the team that restarts cannot be the team that put it out.
- **Dead ball** runs from the ball leaving play, or play being stopped, to the restart kick or
  throw. Moving the ball to where the restart is taken is not play. A goal kick is one stoppage.
- **A whistle with no ball out** is read from the players: they all slow to walking pace and stay
  there.
- **Stoppages shorter than 3 seconds are treated as live play**, unless the ball is seen clearly off
  the pitch.
- **The order of the rules** is part of the method: possession is settled first, then checked, and
  the stoppage rules have the last word.

## 5. Limitations

- **The public score flatters the method.** Several steps were written after we looked at cases
  inside the public five minutes: the touch steps, the goal-kick, restart and placement rules, the
  trusted-touch filter, and the two whistle steps. Each is a general rule with settings from football
  or from other matches, and none refers to a frame of this match, but the cases that prompted them
  are in that window. Public check of the submitted file: spellF1 0.703 (90% range 0.53 to 0.85),
  macroF1 0.916.
- **An independent estimate is lower.** A port of the chain to the seven outside matches (one camera,
  no tags) gets live or dead right on 89% of frames and the team right on 87% of live frames, with
  spellF1 of 0.41 against their labels and 0.53 after merging their possessions under 3 seconds
  (their label changes on every touch). On that data most rule steps move the score by less than
  0.02; the probability model does most of the work.
- **No visible ball.** The ball is seen on about 60% of frames. Without it the team is read from
  movement alone, and that is where most wrong-team frames are.
- **Stoppages with no ball out and no tag** (offside, fouls away from the ball) are found from the
  players slowing down, which starts about a second late and is wrong about one time in ten on
  other matches.
- **The completed-pass rule could not be tested on outside data**, whose possession label follows a
  different convention; there it is slightly negative.
- **Possessions under about 1.5 seconds** are not found; the two-plays rule works against them.
- **Corners and about half the goal kicks**, where neither camera sees the ball.
- **Tracking errors that remain:** a team shows more than 11 players on 6% of frames.

With more time: evidence for the ball-less stretches (the referee's whistle in the audio, the
broadcast cutting away from live play), and a proper held-out test on matches labelled with our
possession convention.

## 6. Reproduction

Python 3.11, with the pinned packages:

```bash
pip install -r requirements-ball-knowers.txt
```

Put the data bundle in `data/` as `data/README.md` describes, then:

```bash
python experiments/submit.py --solution 42 --team Ball_knowers --check-reproducible
```

This writes `evaluation/results/Ball_knowers.csv`, validates it, and rebuilds it from the raw files
with an empty cache to confirm the two are identical. The SHA-256 printed must be the one at the top
of this document. Randomness is seeded. The outside matches are not needed to reproduce the file:
the constants taken from them are written in the code.
