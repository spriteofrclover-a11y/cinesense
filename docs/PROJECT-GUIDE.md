# Project Guide

A plain-language explanation of how CineSense works, what the files do, and what
to say when someone asks a technical question.

Written for Sivakumar, Yeshwant and Siddhardha. Read sections 1 to 7 before the
presentation. Section 9 is for Siddhardha only.

---

## 1. The one-minute explanation

If someone asks "what is your project?", say this:

> We built a movie recommendation website. A user rates a few films, and it
> suggests others they might like. What makes it different is that it uses three
> different ways of deciding, not just one, and it explains why it picked each
> film.
>
> The three ways are: what people with similar taste rated highly, what the film
> itself is like in genre and theme, and how popular the film is. We combine
> those into a single score and rank the catalogue.
>
> The dataset is MovieLens 32 million ratings, and the site is live on Vercel.

That is 45 seconds. Do not add detail unless asked.

---

## 2. How it works, in plain words

Think of asking three different friends for a film suggestion.

**Friend one** has watched and rated hundreds of films, and their taste is close
to yours. If they loved something, you probably will too. This is called
**collaborative filtering**, because it uses the collaboration of many users.

**Friend two** does not know you at all, but knows films very well. If you liked
one science fiction film, they suggest another that shares its genre, its themes
and the words people use to describe it. This is called **content-based
filtering**, because it looks at the content of the film.

**Friend three** just says what is popular and well rated. Not personal, but a
useful safety net when the other two are unsure.

We ask all three, give each one a vote, and combine the votes into one ranking.
Because we kept the votes separate, we can also tell you which friend spoke
loudest for each film. That is where the "because you rated X" line comes from.

**Why three and not one?** We measured it. Any single method did worse than the
combination. One of them, the content-based one, was actually worse than just
showing popular films, because "similar film" and "film you will rate highly" are
not the same question.

---

## 3. What happens when you rate a film

This is the most likely thing you will be asked to explain, so know these steps.

1. **The browser saves your rating.** It goes into the browser's own storage on
   your machine. Nothing is created on the server.
2. **The browser sends all your ratings with each request.** Every time it asks
   for recommendations, it sends the whole list.
3. **The server builds your taste profile.** It adds up the films you liked,
   weighted by how much you liked them, producing one list of numbers that
   represents your taste.
4. **The server scores every film three ways.** Similar users, content, and
   popularity.
5. **It combines the three scores into one.** Different methods produce numbers
   on different scales, so they are converted to a common scale first, otherwise
   the one with the biggest numbers would always win.
6. **It picks the top films and avoids repeating itself.** Without this step a
   list of ten would contain ten almost identical films.
7. **It writes a reason for each film.** The film you rated that is closest to
   this one, plus the genres and themes they share.
8. **The browser draws the cards.** The server never remembers the request.

The important sentence: **the server does not store anything about the user.**
That is why there is no login, no database and no privacy problem.

---

## 4. The file structure

There are two halves, and knowing which is which answers most questions.

- **Offline half** runs once, on our machine. It is slow and uses heavy
  libraries. It produces small files that the app needs.
- **Online half** runs on every request, on the server. It is fast and uses
  almost nothing.

```
configs/base.toml            every setting and weight, in one place
src/cinesense/
    data/                    OFFLINE: download, clean, store the dataset
    features/                OFFLINE: build the numbers that describe films,
                             train the model, measure quality
    recommend/               ONLINE: turn ratings into recommendations
    serving/                 ONLINE: the web server and the website
api/index.py                 the entry point Vercel calls
artifacts/                   the small output files from the offline half
tests/                       automated checks
scripts/                     helper commands
docs/                        screenshots and this guide
```

What each folder does, in one line each:

| Path | What it does |
| --- | --- |
| `data/` | Downloads MovieLens, cleans it, saves it in a fast local format |
| `features/text.py` | Turns genres and tags into numbers that describe a film |
| `features/collab.py` | Learns patterns from the 32 million ratings |
| `features/ranking_eval.py` | Measures how good the ranking is |
| `recommend/content.py` | Finds films similar to ones you liked |
| `recommend/collab.py` | Predicts what you would rate a film |
| `recommend/hybrid.py` | Combines the signals and removes repetition |
| `recommend/explain.py` | Writes the reason in words |
| `serving/app.py` | Starts the web server, loads the model once |
| `serving/routes.py` | The API endpoints |
| `serving/static/` | The website itself: HTML, CSS, JavaScript |
| `artifacts/` | The saved model files, loaded at startup |

Two folders are worth knowing by name:

- `artifacts/` holds the finished model. It is about 32 MB and is committed to
  the repository, so the server never has to train anything.
- `configs/base.toml` holds every number we tuned, such as how much weight each
  of the three signals gets. Nothing is hardcoded in the program.

---

## 5. The technical terms, explained simply

If an examiner uses one of these words, here is what it means and how to answer.

| Term | Plain meaning | One-line answer |
| --- | --- | --- |
| Collaborative filtering | Using other users' ratings | "People who liked what you liked also liked this" |
| Content-based filtering | Using the film's own description | "This film is similar to ones you rated highly" |
| Hybrid | Using both | "We combine them because each alone was worse" |
| TF-IDF | A way to score words by how distinctive they are | "Common words count less, distinctive words count more" |
| Embedding / LSA | A short list of numbers standing in for a description | "We compress a film's description into 96 numbers" |
| SVD | Finding hidden patterns in a big table | "It finds the underlying taste dimensions in the rating table" |
| Cosine similarity | How close two lists of numbers point in the same direction | "A standard way to measure similarity between two films" |
| Cold start | A new user with no history | "They pick films during onboarding, and we start from those" |
| Fold-in | Calculating a user's profile on the spot | "We do not store profiles; we calculate one from your ratings" |
| Regularisation | Stopping the model from overreacting to little data | "So a film with 3 ratings does not get an extreme score" |
| Z-score | Converting numbers to a common scale | "So a 0.8 similarity and a 4.5 star prediction can be added fairly" |

If you remember one analogy, remember the z-score one: you cannot add metres and
feet directly, so you convert them first. Same idea.

---

## 6. The numbers to remember

Learn these six. They come up.

| Number | What it is |
| --- | --- |
| **87,585** | films in the catalogue |
| **32 million** | ratings used, from 200,948 users |
| **0.82** | average error in predicted star rating, out of 5 |
| **11%** | how much better the combined ranking is than one signal alone |
| **192** | automated tests |
| **2.1 seconds** | first load on the live site, under a second after that |

Two more if you want them: 43,129 films have posters and language data, and
1,141 Indian films are reachable through the language filters.

The 0.82 is the one to be careful with. It does not mean "0.82 out of 5 quality".
It means: when the model predicts what you will rate a film, it is wrong by about
0.82 stars on average. Out of 5 stars, that is close.

---

## 7. Questions you will be asked

Short answers. Say the short answer, then stop.

**Why not just recommend popular films?**
We measured that. Popularity alone scored 0.118 on our ranking test, the combined
method scored 0.132. Popularity is still part of the mix, just not the only part.

**Why is the content-based method the weakest?**
Because "this film is similar" is not the same as "you will rate this highly". Two
films can be almost identical in genre and still be loved by different people.

**How do you know it actually works?**
We held back 10 percent of the ratings before training. The model never saw them.
Every number we quote is measured on those held-back ratings.

**What happens if I rate a film one star?**
It is excluded from your recommendations rather than counted against similar
films. We found that treating a low rating as a negative made the suggestions
much worse, because disliking one science fiction film pushed us away from all
science fiction.

**How does a new user with no ratings get anything?**
They pick films they love during onboarding. Those picks become their starting
ratings, and the same code path runs. There is no special case.

**Where is my data stored?**
Only in the browser, on that device. The server receives ratings with each
request and does not keep them.

**Why is the first load slow?**
The server loads the model files into memory, about 32 MB. That costs about two
seconds once, then each request is fast.

**Why Python?**
Every serious data and machine learning library is there, and it was the right
tool for cleaning 32 million rows and training the model.

**Why FastAPI?**
It is lightweight, it validates requests automatically, and it starts fast, which
matters on a serverless host.

**What was the hardest part?**
Make it personal to you. The honest answer for the technical side: making the
three signals comparable, and stopping the model from recommending the same film
ten times in a row.

**What would you improve?**
We cannot read a plot, only genres and tags. And our dataset is Western-heavy, so
we added language filters but films missing from the dataset still cannot be
recommended.

**Did you use AI tools?**
Answer honestly and in your own words. If you did, say how: for example, to
generate and check code faster, then reviewed and tested it. Do not claim you
wrote every line by hand if you did not, and do not apologise for using tools
either.

---

## 8. Who says what

Agree on this beforehand so you do not contradict each other.

**Sivakumar** takes the data and the models:
- how the dataset was cleaned and prepared
- what the three signals are and why three
- how the model was trained and tested
- the numbers

**Yeshwant** takes the application:
- how the API answers a request
- how the website talks to it
- how it was deployed
- what the tests check

**Siddhardha** takes the research:
- why recommenders are studied this way
- what other approaches exist and how they compare
- why we chose this one
- the report and the presentation

When a question lands that is not yours, hand it over by name: "Yeshwant handled
the deployment, he can explain that." That reads as a team, not as three people
who each memorised a third.

---

## 9. For Siddhardha: the research side

The project's research contribution is not a separate experiment. It is the
reasoning that decided what we built. Be ready to walk through it in this order,
because it is the order a reviewer will expect.

**Step 1. Start from the problem, not the algorithm.**
Catalogues are too large to browse, and a recommendation nobody understands is one
nobody acts on. Two failures matter: relevance, and trust. Everything after this
follows from those two.

**Step 2. Survey the three families of methods.**
Content-based, collaborative, and hybrid. For each: how it works, what it needs,
and where it fails. Content-based needs no ratings but cannot tell quality.
Collaborative is strong but useless for a new user. Hybrid exists because each
covers the other's weakness.

**Step 3. Compare the available datasets.**
Netflix Prize (1M ratings, withdrawn), MovieLens (multiple sizes, active),
TMDB (rich metadata, no ratings). MovieLens 32M was chosen for scale and because
it includes tags; TMDB was added for posters and language.

**Step 4. Decide how to measure success.**
This is the point most student projects skip. Prediction accuracy alone is not
enough. A recommender can predict ratings well and still rank badly. So we used
two kinds of measure: error on predicted ratings, and ranking quality at the top
of the list. That decision is what later revealed that content-only was worse
than popularity.

**Step 5. Justify the extra data.**
We did not assume the tag genome helped. We measured it against a signal that was
not part of the model, and it did. Being able to say "we tested whether it was
worth it" is stronger than "we used it because it is available".

**Step 6. Report honestly.**
State the limits: no plot understanding, a Western-skewed catalogue, weak
modelling for rarely rated films, no user accounts. A stated limit reads as
competence. A hidden one reads as an oversight.

If asked what you personally contributed, that chain above is your answer: the
literature, the dataset choice, the metric choice, the justification, and the
documentation.

---

## 10. If you do not know the answer

Do not invent one. Use one of these.

- "I did not work on that part. Sivakumar can explain it."
- "We did not test that, so I would not want to guess."
- "That is a fair limitation, and it is in our future work."

An honest "we did not measure that" is respected. A confident wrong number is
not, and it is the one thing that will be remembered.
