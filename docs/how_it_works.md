# What is this thing? (Explained like you're 5)

Imagine a little toy box with a screen, two buttons and a speaker, sitting on
your bedside table. You press a button to pick your favourite character — like
Bluey — and the box makes up a brand-new bedtime story about them, shows it on
the screen, and reads it out loud in a calm voice until you fall asleep.

The important part: the box never talks to the internet. Not ever. Not once
after it's built. Like a bedtime book, but one that can write itself.

```mermaid
flowchart LR
    A["🧒 Kid presses a button"] --> B["📦 The box<br/>makes up a story"]
    B --> C["🖥️ Shows the words<br/>on its screen"]
    B --> D["🔊 Reads it out<br/>in a calm voice"]
    D --> E["😴 Zzzz..."]
```

## 1. The box is really, really tiny

Inside the box is a computer chip smaller than your fingernail. It's not a big
computer like a laptop. It has:

```mermaid
flowchart TB
    subgraph box["📦 The magic box (M5StickS3)"]
        C["🧠 Tiny brain<br/>(ESP32-S3 chip)"]
        F["📚 Big shelf of stuff<br/>(8 MB flash)"]
        M["🗃️ Middle shelf<br/>(8 MB memory)"]
        T["👛 Tiny pocket<br/>(512 KB fast memory)"]
        S["🖥️ Tiny screen<br/>(smaller than a matchbox)"]
        K["🔘 Buttons"]
        V["🔈 Speaker"]
    end
    C --> F
    C --> M
    C --> T
```

That's why everything has to be small and clever. One whole movie is already
too big for the entire box. A big AI brain? Forget it. So we cheat — cleverly.

## 2. How long is a bedtime story?

Long enough to fall asleep to, short enough to finish:

| | |
|---|---|
| Words | **200–250** |
| Lines (sentences) | about **22** |
| Reading time | about **2 minutes** |
| On the screen | about **3 pages**, turned one line at a time |

## 3. How does the box "invent" a story? It follows a bedtime recipe.

The box can't "think up" stories the way you do. It builds one like a recipe.
Every story walks the same sleepy path, and the box picks each step from a
little pile of choices:

```mermaid
flowchart TB
    O["🌍 Opening<br/>the world + the flavour"] --> S1["🧸 Settle x3<br/>blanket, pillow, pyjamas"]
    S1 --> S2["🌙 Sky x2<br/>stars, moon, quiet night"]
    S2 --> P["😟 A small problem x2<br/>someone can't sleep"]
    P --> H["🤝 A helper<br/>a friend from the same world"]
    H --> T["🔁 Trying x4<br/>help / share / brave / try"]
    T --> W["🤗 All warm again x3"]
    W --> Q["🌫️ Quiet x3<br/>the whole house calms down"]
    Q --> Z["😴 Sleep x2<br/>eyes close"]
    Z --> C["🌍 Closing<br/>everyone asleep, always"]
```

The important bit: the **sleepy ending always comes last**. No matter which
flavour you pick — adventure, funny, kind — the story still ends with everyone
fast asleep. That's what makes it a bedtime box.

It then fills in the blanks, like a colouring book:

> "{hero} carried {thing} to the {place}." becomes

> "**Bluey** carried **a soft teddy** to **the bedroom window**."

```mermaid
flowchart LR
    W["🌍 World<br/>Bluey / Gabby / One /<br/>Amelia / Junie"]
    P["🧩 Thing<br/>teddy / night light /<br/>blanket / storybook"]
    Z["🗺️ Place<br/>bedroom / cubby /<br/>window / garden"]
    W & P & Z --> F["✍️ Fill the blanks"] --> S["📖 A sentence!"]
```

## 4. How does it pick? By rolling fair dice, the same way every time

The box has fair dice inside. Same question, same dice roll, same story. That's
a superpower: if a story is wrong, we can find it again, exactly, until we fix
it.

```mermaid
flowchart LR
    Q["❓ Hero + flavour + a number"] --> D["🎲 Fair dice"]
    D --> R1["😟 which little problem"]
    D --> R2["🧸 which cosy thing"]
    D --> R3["👭 which friend helps"]
```

## 5. Before the box says anything, a Safety Guard checks it

There's a guard standing at a gate inside the box. Every story must pass the
guard's checklist before it can be shown or spoken:

```mermaid
flowchart TB
    S["📖 A story"] --> G{"🛡️ The Guard's checklist"}
    G -->|"has a real ending"| OK
    G -->|"not too long,<br/>not too short<br/>(200-250 words)"| OK
    G -->|"no repeated lines"| OK
    G -->|"only words it can SAY"| OK
    G -->|"right hero, right world"| OK
    OK["✅ OK<br/>show it + say it"]
    G -->|"anything wrong?"| R["🚫 REJECT"]
    R --> F["💾 Read a safe<br/>ready-made story<br/>instead"]
```

The most important check is the **word stamp**: the box only allows words it
knows how to say (722 words today). So "Bluey found a knife" can never get
through, because "knife" has no stamp — even though the sentence shape is
perfect.

## 6. How does it talk? With LEGO voices, not a robot brain

A talking robot brain is WAY too big for our tiny box. So instead a friendly
voice is recorded **before** the box is built: little sound bricks like
*"Bluey carried"*, *"a soft teddy"*, *"to the bedroom window"*.

When the story is ready, the box snaps the right bricks together:

```mermaid
flowchart LR
    C1["🔊 'Bluey carried'"] --- C2["🔊 'a soft teddy'"] --- C3["🔊 'to the bedroom window'"] --- C4["🔊 '.'"]
```

That's about 218 little sound bricks (~4 minutes of recorded voice, ~2 MB). The
bricks fit on the big shelf, and each story borrows about 890 kB of them as it
plays. It sounds like a nice person reading — not a robot.

## 7. Where does the tiny computer "think"? On three shelves

```mermaid
flowchart TB
    A["📚 Big shelf: 8 MB<br/>Kept there: the story recipe book,<br/>the sound bricks, the tiny model<br/>(slow to reach, huge)"]
    B["🗃️ Middle shelf: 8 MB<br/>Kept there: the story being built"]
    C["👛 Tiny pocket: 512 KB<br/>Kept there: what it's thinking<br/>about THIS second (super fast, tiny)"]
```

Everything big lives on the big shelf. The box only holds a tiny bit in its
pocket at once — like reading a book one page at a time instead of carrying the
whole library on your head.

## 8. So where does the "tiny language model" (the AI part) go?

Right now the recipe-picker is just fair dice 🎲 (built and tested over 15,360
stories). Later we can swap the dice for a tiny brain 🧠 — but one that only
**picks the recipe steps, never writes the words**:

```mermaid
flowchart LR
    subgraph today["🗓️ today — built ✅"]
        T1["kid picks Bluey"] --> T2["🎲 dice pick each step"] --> T3["📖 story"]
    end
    subgraph later["🔜 later — optional smarter version"]
        L1["kid picks Bluey"] --> L2["🧠 tiny brain picks<br/>the best steps"] --> L3["📖 story"]
    end
```

That's the trick that keeps everything safe: the brain can't say a bad word
because it doesn't know words — it only knows recipe steps. The guard still
stands at the gate. The voice bricks always fit.

## 9. What's finished and what's not

| | Done? |
|---|---|
| The recipe book + the dice + the guard | ✅ 327 checks green, 15,360 stories |
| Five bedtime worlds (Bluey, Gabby, One, Amelia, Junie) | ✅ in the code |
| Story data for the tiny brain | ✅ 2,880 stories, all different |
| The real box (screen, buttons, speaker) | ❌ needs the actual toy + tools plugged in |
| The ~218 sound bricks | ❌ needs you to read ~218 lines out loud |
| The optional tiny brain | ❌ last, after all of that |

## Grown-up note (the boring part)

The five worlds (Bluey, Gabby's Dollhouse, Numberblocks, Amelia Bedelia, Junie
B. Jones) each belong to the company that owns them. For a toy **you** build
and keep at home, that's fine. It would not be fine to sell it without asking
those companies, so we keep a note of who owns what in the code, so a "for
everyone" version can simply leave those worlds switched off.
