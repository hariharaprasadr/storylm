#!/usr/bin/env python3
"""StoryLM Corpus Builder and Vocabulary Scaler.

Scales the vocabulary of StoryLM to 3.5K words (~3,500 words / 3,530-3,589 total tokens),
incorporates authentic character arcs for:
  - Bluey (creative play, games like Keepy-Uppy, Shadowlands, sisterly dynamics with Bingo, Bandit/Chilli)
  - Gabby (dollhouse delivery, shrinkification, craft room, Cakey sprinkles, CatRat, MerCat, Pillow Cat)
  - Numberblocks (Numberland, arrays, step squads, adding and balancing, Blockzilla, counting down)
  - Amelia Bedelia (hilarious literal interpretations: dusting with powder, dressing chickens,
                   drawing drapes with crayons, pitching tents, delicious homemade lemon pies and sweet apologies)
  - Junie B. Jones (authentic first-grade voice, Lucille & Grace, bus friend Herb, stomach butterflies,
                    schoolyard drama, honest apologies, cozy tuck-ins)

Introduces new templates for:
  - Modes: bedtime, adventure, funny, kindness, + mystery, discovery, celebration, silly
  - Places: backyard, bedroom, cubby, creek, verandah, beach, lounge, playroom, window, craftroom,
            kitchenette, fairygarden, musicroom, garden, stair, numberland, tower, meadow, patternpark,
            kitchen, livingroom, pantry, parlor, orchard, porch, hallway, yard, classroom, playground, schoolbus
  - Themes: help, share, brave, try, + honesty, curiosity, patience, friendship, creativity

Guarantees:
  - Story budget: strictly 201-248 words per story (satisfies 200 min, 250 max)
  - Line budget: 13-30 lines per story (typically 20-23 lines)
  - Line length: <= 16 words per line
  - Closed vocabulary: 100% of generated story words exist in data/vocab.json
  - Vocabulary scaled to ~3.5K words
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

WORD_RE = re.compile(r"[a-z]+")
TAG_RE = re.compile(r"<[^>]+>")


def normalize_words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Canon Series Configuration
# ---------------------------------------------------------------------------

SERIES_CONFIG = [
    {
        "id": "bluey",
        "name": "Bluey",
        "story_name": "Bluey",
        "token": "<CHAR=BLUEY>",
        "description": "a six-year-old blue heeler puppy who loves imaginative family play",
        "rights": "Ludo Studio / BBC Studios (all rights reserved)",
        "default_mode": "bedtime",
        "modes": ["bedtime", "adventure", "funny", "kindness", "mystery", "discovery", "celebration", "silly"],
        "friends": ["Bingo", "Bandit", "Chilli", "Muffin", "Rusty", "Mackenzie"],
        "places": ["backyard", "bedroom", "cubby", "creek", "verandah", "beach", "lounge"],
        "props": [
            "a warm blanket", "a night light", "a bedtime storybook", "a soft teddy",
            "a woolly quilt", "a cosy dressing gown", "a sleepy lullaby", "a red balloon",
            "a magic featherwand", "floppy the bunny", "a bush stick", "a striped beach towel"
        ]
    },
    {
        "id": "gabby",
        "name": "Gabby",
        "story_name": "Gabby",
        "token": "<CHAR=GABBY>",
        "description": "a girl who shrinks down with magical cat ears to explore her dollhouse",
        "rights": "DreamWorks Animation (all rights reserved)",
        "default_mode": "bedtime",
        "modes": ["bedtime", "adventure", "funny", "kindness", "mystery", "discovery", "celebration", "silly"],
        "friends": ["Pandy Paws", "Cakey", "CatRat", "MerCat", "Pillow Cat", "Baby Box"],
        "places": ["bedroom", "playroom", "window", "craftroom", "kitchenette", "fairygarden", "musicroom"],
        "props": [
            "a fluffy pillow", "a glow star", "a cosy blanket", "a music box",
            "a sparkly quilt", "a tiny lantern", "a tiny teapot", "a sprinkle shaker",
            "a glitter ribbon", "a magical delivery box", "a bubble bottle", "a storybook"
        ]
    },
    {
        "id": "numberblocks",
        "name": "One",
        "story_name": "One",
        "token": "<CHAR=ONE>",
        "description": "a curious little red number block discovering mathematics and friendship",
        "rights": "Alphablocks Ltd / Blue Zoo (all rights reserved)",
        "default_mode": "bedtime",
        "modes": ["bedtime", "adventure", "funny", "kindness", "mystery", "discovery", "celebration", "silly"],
        "friends": ["Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"],
        "places": ["playroom", "garden", "stair", "numberland", "tower", "meadow", "patternpark"],
        "props": [
            "a soft blanket", "a night light", "a star chart", "a cuddly teddy",
            "a knitted quilt", "a fluffy pillow", "a bedtime rhyme", "a magic mirror",
            "a stepping block", "a balance beam", "a geometric cube", "a shiny number badge"
        ]
    },
    {
        "id": "amelia",
        "name": "Amelia Bedelia",
        "story_name": "Amelia",
        "token": "<CHAR=AMELIA>",
        "description": "an earnest housekeeper who takes every idiom literally, then bakes sweet pies",
        "rights": "Peggy Parish estate / HarperCollins (all rights reserved)",
        "default_mode": "bedtime",
        "modes": ["bedtime", "adventure", "funny", "kindness", "mystery", "discovery", "celebration", "silly"],
        "friends": ["Mr. Rogers", "Mrs. Rogers", "Alcolu", "Mrs. Wragge"],
        "places": ["kitchen", "garden", "livingroom", "pantry", "parlor", "orchard", "porch"],
        "props": [
            "a warm blanket", "a hot water bottle", "a bedtime book", "a night light",
            "a patchwork quilt", "a lavender cushion", "a mug of cocoa", "a fresh lemon pie",
            "a feather duster", "a tin of dusting powder", "a wooden rolling pin", "a calico apron"
        ]
    },
    {
        "id": "junie",
        "name": "Junie B. Jones",
        "story_name": "Junie",
        "token": "<CHAR=JUNIE>",
        "description": "an outspoken first-grader who navigates school adventures with dramatic charm",
        "rights": "Barbara Park estate / Random House (all rights reserved)",
        "default_mode": "bedtime",
        "modes": ["bedtime", "adventure", "funny", "kindness", "mystery", "discovery", "celebration", "silly"],
        "friends": ["Lucille", "Grace", "Herb", "Mrs.", "Ollie", "Mr. Scary"],
        "places": ["bedroom", "hallway", "yard", "classroom", "playground", "schoolbus", "cafeteria"],
        "props": [
            "a teddy bear", "a night light", "a bedtime book", "a fluffy blanket",
            "a stripy quilt", "a woolly jumper", "a bedtime rhyme", "philip johnny bob",
            "a purple crayon", "a shiny shoe", "a funny pair of glasses", "a pink pencil"
        ]
    }
]

MODES = ["bedtime", "adventure", "funny", "kindness", "mystery", "discovery", "celebration", "silly"]
THEMES = ["help", "share", "brave", "try", "honesty", "curiosity", "patience", "friendship", "creativity"]


# ---------------------------------------------------------------------------
# Template Banks for Story Phases
# ---------------------------------------------------------------------------

OPENINGS = {
    # Bluey
    ("bluey", "bedtime"): [
        "The sun was dipping low when {name} carried {prop} across the {place}.",
        "Dusk was falling when {name} brought {prop} inside the peaceful {place}.",
        "Evening settled quietly over the house as {name} stood in the {place}."
    ],
    ("bluey", "adventure"): [
        "After a grand afternoon playing outdoors, {name} carried {prop} to the {place}.",
        "Their energetic backyard journey finished just as {name} walked into the {place}.",
        "A lively game of shadow exploration wrapped up when {name} entered the {place}."
    ],
    ("bluey", "funny"): [
        "A delightfully silly game had just finished when {name} brought {prop} into the {place}.",
        "{name} giggled softly while carrying {prop} across the {place} for bedtime.",
        "Chuckling over a funny hotel gag, {name} brought {prop} into the cosy {place}."
    ],
    ("bluey", "kindness"): [
        "With a caring smile, {name} carried {prop} into the {place} to comfort {friend}.",
        "Thinking of a gentle gesture, {name} brought {prop} straight to the quiet {place}.",
        "{name} wanted everyone to feel comforted as evening arrived at the {place}."
    ],
    ("bluey", "mystery"): [
        "Searching for missing clues in the twilight, {name} carried {prop} through the {place}.",
        "A charming evening puzzle began right as {name} arrived at the quiet {place}.",
        "Investigating gentle shadows, {name} brought {prop} to inspect the cosy {place}."
    ],
    ("bluey", "discovery"): [
        "Finding a wonderful secret near the {place}, {name} held {prop} with wonder.",
        "{name} noticed a tiny magical detail while carrying {prop} to the {place}.",
        "A peaceful discovery surprised {name} while stepping quietly across the {place}."
    ],
    ("bluey", "celebration"): [
        "A delightful family festival was ending when {name} placed {prop} in the {place}.",
        "After cheerful dancing and clapping, {name} carried {prop} into the sleepy {place}.",
        "The joyous holiday party calmed down as {name} reached the peaceful {place}."
    ],
    ("bluey", "silly"): [
        "Playing a ridiculous backwards walking game, {name} carried {prop} to the {place}.",
        "{name} hopped on one paw and brought {prop} into the {place}.",
        "Wiggling with sleepy puppy giggles, {name} set {prop} down in the {place}."
    ],

    # Gabby
    ("gabby", "bedtime"): [
        "Dollhouse deliveries were done for the day when {name} brought {prop} to the {place}.",
        "The dollhouse lights flickered softly as {name} carried {prop} into the {place}.",
        "Twilight purred across the rooms when {name} stepped into the {place}."
    ],
    ("gabby", "adventure"): [
        "Shrinkification had brought a fantastic quest before {name} carried {prop} to the {place}.",
        "After exploring every dollhouse hallway, {name} carried {prop} to the {place}.",
        "Their miniature dollhouse expedition finished as {name} brought {prop} to the {place}."
    ],
    ("gabby", "funny"): [
        "A sprinkle explosion had made everyone laugh before {name} entered the {place}.",
        "Giggling with {friend}, {name} carried {prop} into the {place}.",
        "CatRat had tried a cheeky prank before {name} brought {prop} to the {place}."
    ],
    ("gabby", "kindness"): [
        "Sharing a warm cat hug, {name} brought {prop} to comfort {friend} in the {place}.",
        "{name} made sure all the Gabby Cats were comfy in the {place}.",
        "With gentle paws and a warm heart, {name} carried {prop} into the {place}."
    ],
    ("gabby", "mystery"): [
        "A mysterious sparkle in the corner led {name} and {friend} into the {place}.",
        "Searching for a missing glitter ribbon, {name} carried {prop} to the {place}.",
        "A gentle dollhouse riddle was waiting when {name} entered the {place}."
    ],
    ("gabby", "discovery"): [
        "Unboxing a magical delivery, {name} carried {prop} to show {friend} in the {place}.",
        "{name} discovered a glowing craft treasure right inside the quiet {place}.",
        "A surprising tiny invention delighted {name} in the peaceful {place}."
    ],
    ("gabby", "celebration"): [
        "The great sprinkle party was winding down as {name} carried {prop} to the {place}.",
        "After a cheerful spa celebration, {name} brought {prop} into the quiet {place}.",
        "The dollhouse festival ended gently when {name} walked into the {place}."
    ],
    ("gabby", "silly"): [
        "Dancing a silly kitty waltz, {name} carried {prop} across the {place}.",
        "{name} laughed at playful cat whiskers while bringing {prop} to the {place}.",
        "Making playful meow sounds, {name} nestled {prop} in the quiet {place}."
    ],

    # Numberblocks
    ("numberblocks", "bedtime"): [
        "The stars above Numberland began to blink when {name} carried {prop} to the {place}.",
        "One red block glowed warmly as {name} rested beside {prop} in the {place}.",
        "Evening numbers gathered quietly while {name} brought {prop} into the {place}."
    ],
    ("numberblocks", "adventure"): [
        "After climbing the highest number stairs, {name} brought {prop} into the {place}.",
        "A great expedition across the geometric hills brought {name} into the {place}.",
        "Exploring number patterns all day, {name} carried {prop} to the quiet {place}."
    ],
    ("numberblocks", "funny"): [
        "The silly Two clown shoes made everyone chuckle before {name} reached the {place}.",
        "Juggling three round apples, {name} giggled and carried {prop} to the {place}.",
        "A funny number tumble ended with smiles when {name} brought {prop} to the {place}."
    ],
    ("numberblocks", "kindness"): [
        "{name} gently stacked beside {friend} to share {prop} in the {place}.",
        "Showing that every block counts, {name} brought {prop} into the {place}.",
        "{name} helped {friend} balance upright while carrying {prop} to the {place}."
    ],
    ("numberblocks", "mystery"): [
        "Counting stepping stones in the twilight, {name} carried {prop} across the {place}.",
        "A curious pattern of shadows appeared when {name} entered the {place}.",
        "Solving a missing block puzzle, {name} brought {prop} into the {place}."
    ],
    ("numberblocks", "discovery"): [
        "{name} discovered a brand new array shape while carrying {prop} to the {place}.",
        "Adding up shiny stars revealed a wonder when {name} reached the {place}.",
        "A lovely geometric symmetry surprised {name} inside the quiet {place}."
    ],
    ("numberblocks", "celebration"): [
        "The big step squad parade concluded happily when {name} reached the {place}.",
        "After celebrating ten fantastic leaps, {name} carried {prop} into the {place}.",
        "Numberland cheered with joy before {name} rested beside {prop} in the {place}."
    ],
    ("numberblocks", "silly"): [
        "Spinning in circles like a rolling block, {name} carried {prop} into the {place}.",
        "{name} made funny square faces while bringing {prop} into the {place}.",
        "Hopping on corners with a grin, {name} carried {prop} across the {place}."
    ],

    # Amelia Bedelia
    ("amelia", "bedtime"): [
        "The day was done when {name} carried {prop} into the cozy {place}.",
        "{name} finished her evening duties and brought {prop} straight to the {place}.",
        "Dusk fell across the quiet farmhouse as {name} stepped into the {place}."
    ],
    ("amelia", "adventure"): [
        "After an eventful outdoor stroll, {name} carried {prop} into the {place}.",
        "Her lively afternoon chores finished when {name} reached the peaceful {place}.",
        "Following directions with great enthusiasm, {name} brought {prop} to the {place}."
    ],
    ("amelia", "funny"): [
        "{name} chuckled over another unusual instruction while carrying {prop} to the {place}.",
        "Wiping a spot of flour from her cheek, {name} brought {prop} into the {place}.",
        "A delightful misunderstanding had ended cheerfully when {name} entered the {place}."
    ],
    ("amelia", "kindness"): [
        "Eager to make everyone comfortable, {name} brought {prop} into the {place}.",
        "{name} wanted {friend} to feel thoroughly cherished in the {place}.",
        "With a generous spirit, {name} carried {prop} straight to the quiet {place}."
    ],
    ("amelia", "mystery"): [
        "Wondering what a curious note meant, {name} carried {prop} into the {place}.",
        "{name} searched carefully for the missing recipe inside the quiet {place}.",
        "A charming household puzzle caught {name} while walking through the {place}."
    ],
    ("amelia", "discovery"): [
        "{name} found an old handwritten recipe book while carrying {prop} to the {place}.",
        "A lovely basket of fresh cinnamon apples greeted {name} in the {place}.",
        "Discovering a warm spot beside the hearth, {name} set {prop} down in the {place}."
    ],
    ("amelia", "celebration"): [
        "A festive evening supper concluded happily when {name} entered the {place}.",
        "Having baked a magnificent dessert, {name} carried {prop} to the quiet {place}.",
        "Everyone had praised her wonderful pie before {name} reached the {place}."
    ],
    ("amelia", "silly"): [
        "{name} giggled about dusting furniture with talcum powder while entering the {place}.",
        "Remembering she had dressed the chicken in doll clothes, {name} entered the {place}.",
        "Smiling at her own funny interpretation, {name} carried {prop} into the {place}."
    ],

    # Junie B. Jones
    ("junie", "bedtime"): [
        "The long day was over when {name} carried {prop} into the {place}.",
        "{name} put on her favourite socks and brought {prop} into the {place}.",
        "Night was coming fast when {name} marched into the {place}."
    ],
    ("junie", "adventure"): [
        "After an exciting quest around the neighbourhood, {name} brought {prop} to the {place}.",
        "Her big expedition had finally wrapped up when {name} entered the {place}.",
        "Running in from the outdoors, {name} carried {prop} straight to the {place}."
    ],
    ("junie", "funny"): [
        "{name} had giggled until her sides ached before bringing {prop} to the {place}.",
        "Remembering a hilarious joke from school, {name} carried {prop} to the {place}.",
        "A funny commotion had ended with smiles when {name} stepped into the {place}."
    ],
    ("junie", "kindness"): [
        "{name} wanted to be the very nicest friend, so she brought {prop} to the {place}.",
        "With a caring heart, {name} carried {prop} in to share with {friend} in the {place}.",
        "Feeling especially sweet and generous, {name} brought {prop} to the quiet {place}."
    ],
    ("junie", "mystery"): [
        "Trying to solve the case of the lost mitten, {name} brought {prop} to the {place}.",
        "{name} tiptoed like a secret detective while carrying {prop} through the {place}.",
        "A mysterious school mystery needed solving before {name} reached the {place}."
    ],
    ("junie", "discovery"): [
        "{name} discovered the best shiny treasure before carrying {prop} to the {place}.",
        "Finding something amazing in her pocket, {name} brought {prop} into the {place}.",
        "A surprising discovery made {name} beam with pride inside the {place}."
    ],
    ("junie", "celebration"): [
        "The classroom party had been a total triumph before {name} reached the {place}.",
        "After shouting hooray for winning a fun game, {name} carried {prop} to the {place}.",
        "A lovely celebration finished as {name} settled down inside the {place}."
    ],
    ("junie", "silly"): [
        "Making funny animal noises for {friend}, {name} carried {prop} into the {place}.",
        "{name} skipped backwards with a giggle while bringing {prop} into the {place}.",
        "Tickled pink by pure silliness, {name} carried {prop} into the {place}."
    ],
}

SETTLE_BEATS = [
    "{name} pulled the covers up and got thoroughly cosy in the quiet space.",
    "{name} smoothed the warm blanket over their toes and smiled gently.",
    "{name} shook out the pillow and punched it into a comfy shape.",
    "{name} arranged the soft cushions neatly along the corner.",
    "{name} kicked off the daytime shoes and slipped into soft slippers.",
    "{name} smoothed out the wrinkles in the quilt and rested their head.",
    "{name} let the busy events of the day fade away like mist.",
    "{name} leaned back against the plush bolster with a contented sigh.",
    "{name} tucked the corners in snug and breathed a peaceful breath."
]

SKY_BEATS = [
    "Outside, the evening sky turned from dusky purple to deep velvety blue.",
    "A pale silver moon climbed above the roof, casting quiet silver ribbons.",
    "The twilight breeze whispered softly through the leaves of nearby branches.",
    "Tiny stars began to sparkle like glittering lanterns across the heavens.",
    "The shadows in the garden grew long and gentle beneath the starlight."
]

CHARACTER_PROBLEMS = {
    "bluey": {
        "help": [
            "{friend} was having trouble untangling the big pile of dress-up blankets.",
            "{friend} felt a bit nervous about the shadowy corner in the hallway."
        ],
        "share": [
            "Both puppies wanted to play with the shiny red balloon at the same time.",
            "There was only one royal crown left in the dress-up box for their game."
        ],
        "brave": [
            "A sudden rattling branch outside sounded like an imaginary backyard monster.",
            "The dark hallway looked long and mysterious in the nighttime shadows."
        ],
        "try": [
            "Keeping the balloon up in the air without touching the carpet was hard.",
            "Balancing the tall cardboard cubby wall took careful coordination and patience."
        ],
        "honesty": [
            "An accidental bump had knocked a stack of building blocks over quietly.",
            "Someone had hidden the favourite book under the couch and felt shy to admit it."
        ],
        "curiosity": [
            "A tiny glowing firefly was fluttering near the windowsill, sparking wonder.",
            "They wondered where the daytime birds went to sleep when the moon came up."
        ],
        "patience": [
            "Waiting for their turn in the hotel game felt like a very long time.",
            "The bedtime timer had not chimed yet, but everyone felt eager to finish."
        ],
        "friendship": [
            "{friend} was feeling a bit left out of the imaginative tea party game.",
            "A small disagreement about the rules of taxi had made {friend} quiet."
        ],
        "creativity": [
            "They needed a clever way to turn two pillows and a towel into an aeroplane.",
            "Inventing a new bedtime dance required matching silly puppy paws together."
        ]
    },
    "gabby": {
        "help": [
            "Cakey had spilled a bowl of rainbow sprinkles across the dollhouse counter.",
            "MerCat needed an extra pair of paws to hold the glowing bath bubble tube."
        ],
        "share": [
            "CatRat had scooped up all the glitter ribbons into his sneaky little pouch.",
            "There was only one sparkly tiara for the dollhouse dress-up party."
        ],
        "brave": [
            "The dollhouse elevator made a surprising squeak when the lights dimmed down.",
            "A strange rustle in the fairy garden made {friend} freeze with wide eyes."
        ],
        "try": [
            "Folding the tiny paper origami kitten ears was fiddly for little fingers.",
            "Mixing the perfect sleep potion required stirring three exact bubbly drops."
        ],
        "honesty": [
            "CatRat accidentally took a shiny spoon without asking and felt guilty.",
            "A glitter tube had leaked onto the carpet, and someone had to speak up."
        ],
        "curiosity": [
            "A mysterious Meow Meow delivery package sat waiting with a secret bow.",
            "They wondered how the dollhouse clockwork gears ticked so rhythmically."
        ],
        "patience": [
            "The freshly baked sprinkle muffins needed ten minutes to cool down.",
            "Waiting for the magic potion bubbles to settle took calm and quiet waiting."
        ],
        "friendship": [
            "Baby Box was feeling disappointed because her paper windmill would not spin.",
            "Pillow Cat wanted a bedtime story, but everyone else had been too busy."
        ],
        "creativity": [
            "They wanted to build a miniature puppet theatre using empty delivery boxes.",
            "Designing a brand new cat song meant blending chimes and soft bell tones."
        ]
    },
    "numberblocks": {
        "help": [
            "{friend} was wobbling on the high stepping stone and needed support.",
            "A tall number tower was leaning precariously toward the grassy lawn."
        ],
        "share": [
            "Two blocks both wanted the top spot on the victory podium array.",
            "There were only six shiny badges to distribute among the number friends."
        ],
        "brave": [
            "Blockzilla the giant was stomping nearby, looking for larger numbers to count.",
            "Stepping through the dark number portal seemed scary without a buddy."
        ],
        "try": [
            "Figuring out how to combine into a solid square array was tricky.",
            "Finding the exact blocks that add up to ten took several thoughtful guesses."
        ],
        "honesty": [
            "A miscounted sum had led the team down the wrong winding stepping path.",
            "Someone had claimed to count twelve stars when they only counted seven."
        ],
        "curiosity": [
            "A mirror puzzle showed reflections that split blocks into fascinating halves.",
            "They wondered if numbers could keep going forever past a hundred."
        ],
        "patience": [
            "Stacking eight smooth blocks into an upright column took steady hands.",
            "Waiting for each block to take their numbered place in line tested focus."
        ],
        "friendship": [
            "{name} felt small and alone beside the giant double-digit number towers.",
            "{friend} was having trouble finding a partner for the number duo dance."
        ],
        "creativity": [
            "They wanted to arrange themselves into a colorful geometric rainbow bridge.",
            "Inventing a new counting pattern turned mathematics into lively music."
        ]
    },
    "amelia": {
        "help": [
            "The note said to dust the furniture, so {name} showered it with sweet powder.",
            "Asked to dress the chicken for dinner, {name} had put miniature socks on it."
        ],
        "share": [
            "There was only one jar of spiced blackberry jam left in the dark pantry.",
            "Mr. and Mrs. Rogers both fancied the last warm slice of fresh custard pie."
        ],
        "brave": [
            "A loud clatter in the dark pantry sounded like an unexpected visitor.",
            "The grandfather clock in the parlor chimed ten deep, booming strikes."
        ],
        "try": [
            "Rolling out the pie dough into a perfect golden circle took multiple attempts.",
            "Drawing the drapes with crayons onto white paper took great artistic focus."
        ],
        "honesty": [
            "{name} realized she had thrown the tent when told to pitch it on the lawn.",
            "{name} admitted she had put the lights out onto the laundry clothesline."
        ],
        "curiosity": [
            "An unfamiliar herb jar in the kitchen cupboard smelled of cloves and mint.",
            "{name} wondered why people said they hit the hay when bedtime arrived."
        ],
        "patience": [
            "The lemon meringue pie needed forty minutes in the oven to turn golden brown.",
            "Waiting for bread dough to rise under a damp cloth tested eager appetites."
        ],
        "friendship": [
            "Mrs. Rogers looked worn out from town and needed comforting hospitality.",
            "Mr. Rogers was puzzled by his powdered armchair and needed gentle cheering."
        ],
        "creativity": [
            "Instead of ordinary pudding, {name} baked an astonishing blueberry tart.",
            "{name} used colorful ribbon scraps to label all the spice canisters."
        ]
    },
    "junie": {
        "help": [
            "{friend} had dropped her glitter pencil case and crayons rolled under desks.",
            "Her best shoe buckle came unfastened, making walking very awkward."
        ],
        "share": [
            "Both {name} and {friend} wanted the brand-new purple crayon at recess.",
            "There was only one shiny strawberry sticker left on the teacher chart."
        ],
        "brave": [
            "The school hallway was completely empty and echoing with mysterious footsteps.",
            "Going to the principal office to deliver a note made her stomach flutter."
        ],
        "try": [
            "Tying a double knot in her sneaker shoelaces proved very stubborn.",
            "Cutting a straight paper snowflake with safety scissors was tricky business."
        ],
        "honesty": [
            "{name} had accidentally scribbled over a notebook page and felt guilty.",
            "{name} confessed she had hidden Philip Johnny Bob inside her book bag."
        ],
        "curiosity": [
            "A strange cardboard box marked supplies was sitting beside the door.",
            "{name} wondered what the grown-up teachers did after the bell rang."
        ],
        "patience": [
            "Standing in line for the water fountain was hard when her throat was dry.",
            "Waiting for afternoon story hour felt like a million hundred hours."
        ],
        "friendship": [
            "{friend} was having a grouchy day and would not play tag on the grass.",
            "A little disagreement about whose turn it was made recess uncomfortable."
        ],
        "creativity": [
            "They decided to draw an enormous chalk castle across the pavement.",
            "Making up a secret rhyming code made ordinary words sound magical."
        ]
    }
}

HELPER_BEATS = [
    "{friend} noticed the dilemma and stepped right in with a warm, caring smile.",
    "{friend} came over quickly and offered a helping hand without hesitation.",
    "Seeing what was needed, {friend} joined in with cheerful enthusiasm.",
    "{friend} offered a clever idea and encouraged everyone to work as a team.",
    "With a gentle voice, {friend} showed that two heads were better than one."
]

TRY_BEATS = {
    "help": [
        "Together they gathered the scattered pieces and sorted them by size.",
        "They took turns holding the edges steady so nothing slipped away.",
        "With teamwork and care, every little detail fell smoothly into place.",
        "They double-checked their work and gave each other an appreciative nod."
    ],
    "share": [
        "They decided to split the turns equally by counting to twenty.",
        "First one friend enjoyed a turn while the other cheered happily.",
        "Then they swapped smoothly and shared the fun without any fuss.",
        "Sharing made the game twice as delightful as playing alone."
    ],
    "brave": [
        "They held hands tightly and took a deep, steadying breath together.",
        "Turning on the little night light revealed only ordinary peaceful objects.",
        "The scary shadow was just a harmless jacket draped across a chair.",
        "Realising they were safe, their courage blossomed warm and strong."
    ],
    "try": [
        "They paused to look closely at what had gone wrong on the first attempt.",
        "With patient fingers, they adjusted the angle and tried once more.",
        "Practice made each step smoother and much easier than before.",
        "Their perseverance paid off as the project came together beautifully."
    ],
    "honesty": [
        "Speaking in a quiet, honest voice, the truth was shared without excuses.",
        "The gentle explanation cleared away the heavy cloud of worry immediately.",
        "Apologies were warmly given and accepted with understanding hugs.",
        "Honesty brought a wonderful lightness that made everyone breathe easily."
    ],
    "curiosity": [
        "They leaned in close and examined the tiny details with gentle wonder.",
        "Asking thoughtful questions helped unlock the mystery step by step.",
        "Every clue they discovered revealed how wonderfully things fit together.",
        "Their curiosity had transformed a simple moment into a marvelous adventure."
    ],
    "patience": [
        "They sat side by side and counted slow breaths to keep calm.",
        "Watching the slow clock hands taught them that good things take time.",
        "Without rushing, the waiting became peaceful rather than frustrating.",
        "When the moment finally arrived, their patience had made it truly special."
    ],
    "friendship": [
        "They listened carefully to each other and validated how they felt.",
        "Offering a kind compliment brought back genuine smiles and giggles.",
        "They sealed their bond with a warm embrace and mutual understanding.",
        "True companionship proved to be the most comforting gift of the evening."
    ],
    "creativity": [
        "They gathered simple everyday materials and imagined endless possibilities.",
        "Combining their imaginations turned an ordinary corner into an enchanted world.",
        "Colors and shapes blended together into a uniquely inspiring masterpiece.",
        "Their playful invention brought wonder and warmth to everyone present."
    ]
}

WARM_BEATS = {
    "bluey": [
        "The problem was completely solved, and both puppies smiled warmly.",
        "{friend} snuggled close, safe and content in the familiar room.",
        "A gentle wave of relief washed over them as they hugged goodbye."
    ],
    "gabby": [
        "The dollhouse glowed with warm cozy light, and all was right again.",
        "Cakey purred with delight, and MerCat blew a sleepy shiny bubble.",
        "Everyone shared a warm cat-tastic hug before tucking in for the night."
    ],
    "numberblocks": [
        "Every single block stood in its rightful place, balanced and peaceful.",
        "The numbers clicked together harmoniously, content with their achievements.",
        "One and {friend} beamed with pride as Numberland settled into stillness."
    ],
    "amelia": [
        "A warm slice of freshly baked lemon pie made everyone smile with joy.",
        "Mrs. Rogers chuckled fondly and thanked {name} for her good heart.",
        "The warm kitchen aroma of cinnamon and sugar wrapped them in comfort."
    ],
    "junie": [
        "The troublesome mix-up was fixed, and {name} felt proud and relieved.",
        "{friend} whispered that they were bestest friends forever and ever.",
        "Her stomach butterflies flew away, leaving only warm and happy feelings."
    ]
}

QUIET_BEATS = [
    "The clock in the hall ticked on, slow, steady, and soothing.",
    "A soft hush descended across the room, wrapping everything in quiet.",
    "Nothing moved now except the moonlight creeping gently across the floor.",
    "The nighttime shadows rested calm and still against the tranquil walls.",
    "A peaceful silence settled over the corners of the sleepy room.",
    "The quiet outdoors slept soundly under the starry velvet sky."
]

SLEEP_BEATS = [
    "Heavy eyelids fluttered shut as sleepiness took over at last.",
    "Slow, rhythmic breathing filled the tranquil room with peace."
]

CLOSINGS = {
    ("bluey", "bedtime"): "Bluey and {friend} slept on, drifting into wonderful puppy dreams.",
    ("bluey", "adventure"): "Bluey and {friend} slept deeply, resting for tomorrow exciting quests.",
    ("bluey", "funny"): "Bluey and {friend} slept peacefully, with gentle smiles on their faces.",
    ("bluey", "kindness"): "Bluey and {friend} slept warmly, wrapped in safety and deep love.",
    ("bluey", "mystery"): "Bluey and {friend} slept on, while the moon watched over their slumber.",
    ("bluey", "discovery"): "Bluey and {friend} slept soundly, dreaming of stars and peaceful creeks.",
    ("bluey", "celebration"): "Bluey and {friend} slept happily, tired from a festival of joy.",
    ("bluey", "silly"): "Bluey and {friend} slept snugly, giggles melting into silent dreams.",

    ("gabby", "bedtime"): "Gabby and {friend} slept on, cozy inside their magical dollhouse.",
    ("gabby", "adventure"): "Gabby and {friend} slept deeply, dreaming of miniature dollhouse wonders.",
    ("gabby", "funny"): "Gabby and {friend} slept sweetly, smiling at the memory of silly pranks.",
    ("gabby", "kindness"): "Gabby and {friend} slept warmly, purring softly into the quiet night.",
    ("gabby", "mystery"): "Gabby and {friend} slept peacefully under the gentle glow of dollhouse stars.",
    ("gabby", "discovery"): "Gabby and {friend} slept soundly, resting after their wonderful discoveries.",
    ("gabby", "celebration"): "Gabby and {friend} slept comfortably, dreaming of sprinkles and celebrations.",
    ("gabby", "silly"): "Gabby and {friend} slept snugly, wrapped in cat hugs and sleepy songs.",

    ("numberblocks", "bedtime"): "One and {friend} rested quietly, counting down to zero in their dreams.",
    ("numberblocks", "adventure"): "One and {friend} slept soundly, lined up in neat and peaceful arrays.",
    ("numberblocks", "funny"): "One and {friend} rested warmly, with happy thoughts of playful numbers.",
    ("numberblocks", "kindness"): "One and {friend} slept together, secure in the strength of their teamwork.",
    ("numberblocks", "mystery"): "One and {friend} slept deeply, under the quiet geometry of night.",
    ("numberblocks", "discovery"): "One and {friend} rested calmly, dreaming of infinite mathematical wonders.",
    ("numberblocks", "celebration"): "One and {friend} slept comfortably, celebrated by every block in Numberland.",
    ("numberblocks", "silly"): "One and {friend} slept snugly, resting their tired corners until dawn.",

    ("amelia", "bedtime"): "Amelia Bedelia slept peacefully, having done her very best all day long.",
    ("amelia", "adventure"): "Amelia Bedelia rested deeply, dreamily content in her cosy farmhouse room.",
    ("amelia", "funny"): "Amelia Bedelia slept comfortably, smiling in her sleep at the day gags.",
    ("amelia", "kindness"): "Amelia Bedelia slept warmly, knowing she had brought comfort to all.",
    ("amelia", "mystery"): "Amelia Bedelia rested quietly, with the sweet scent of lemon lingering near.",
    ("amelia", "discovery"): "Amelia Bedelia slept soundly, tucked under her beloved patchwork quilt.",
    ("amelia", "celebration"): "Amelia Bedelia rested warmly, appreciated by everyone she loved.",
    ("amelia", "silly"): "Amelia Bedelia slept snugly, ready for another delightful morning tomorrow.",

    ("junie", "bedtime"): "Junie B. and {friend} slept on, hugging their stuffed toys tightly.",
    ("junie", "adventure"): "Junie B. and {friend} slept deeply, resting their tired running shoes.",
    ("junie", "funny"): "Junie B. and {friend} rested peacefully, all out of words for the day.",
    ("junie", "kindness"): "Junie B. and {friend} slept warmly, bestest friends forever under the covers.",
    ("junie", "mystery"): "Junie B. and {friend} slept soundly, while quiet night settled in the hall.",
    ("junie", "discovery"): "Junie B. and {friend} rested sweetly, dreaming of shiny schoolyard wonders.",
    ("junie", "celebration"): "Junie B. and {friend} slept comfortably, tuckered out from all the party cheers.",
    ("junie", "silly"): "Junie B. and {friend} slept snugly, snuggled down into the softest pillows."
}


# ---------------------------------------------------------------------------
# Detail beats: spend the rest of the vocabulary
# ---------------------------------------------------------------------------
# The authored banks above use roughly a third of the token table, so the rest of
# the words would sit in the tied input/output table with no gradient at all.
# These pools place every remaining word in a frame that is grammatical for its
# word class, which buys both vocabulary coverage and wording variety.

# Verbs that do not take a direct object; they get their own frames.
INTRANSITIVE_VERBS = {
    "appear", "arrive", "babble", "bask", "begin", "belong", "bloom", "blush",
    "bounce", "breathe", "burst", "buzz", "celebrate", "chatter", "chime", "clap",
    "climb", "come", "creep", "crept", "crouch", "crumble", "dance", "dangle",
    "dawn", "depart", "descend", "disappear", "doze", "drift", "drip", "droop",
    "dwell", "emerge", "escape", "evaporate", "exist", "fade", "faint", "fall",
    "falter", "flee", "float", "flock", "flow", "flutter", "foam", "gallop",
    "gasp", "gaze", "glance", "gleam", "glide", "glimmer", "glisten", "glow",
    "go", "growl", "grumble", "gurgle", "gush", "happen", "hasten", "hover",
    "howl", "huddle", "hum", "hurry", "jog", "jump", "lag", "land", "lean",
    "leap", "linger", "limp", "lope", "lounge", "lull", "lurch", "lurk",
    "march", "meander", "melt", "murmur", "nestle", "nod", "ooze", "pause",
    "plod", "plunge", "pounce", "pour", "prowl", "purr", "quiver", "ramble",
    "reach", "reappear", "recede", "remain", "reside", "rest", "rise", "roam",
    "roar", "romp", "roost", "run", "rush", "saunter", "scamper", "scatter",
    "scurry", "seem", "seep", "settle", "shimmer", "shine", "shiver", "shudder",
    "sigh", "sit", "skate", "skim", "sleep", "slip", "slumber", "smile",
    "snore", "snuggle", "soar", "sob", "sparkle", "sprawl", "sprint", "sprout",
    "stagger", "stand", "stare", "stay", "strut", "stumble", "surge", "sway",
    "sweep", "swim", "swirl", "toddle", "totter", "travel", "tremble", "trot",
    "trudge", "twinkle", "vanish", "wade", "waft", "wait", "wake", "walk",
    "wander", "wane", "wave", "weep", "wend", "wobble", "yawn",
}

# Words that cannot carry a slot sensibly (function words, bare inflections).
NOT_USABLE = {
    "almost", "behind", "beyond", "either", "enough", "even", "half", "kept",
    "knew", "later", "longer", "middle", "might", "neither", "next", "no",
    "none", "oh", "quite", "somehow", "somewhere", "third", "thought", "uh",
    "well", "built", "dug", "laid", "lit", "sat", "sang", "sank", "slid",
    "spoke", "swung", "taken", "worn", "born", "clad", "gone", "done",
    # degree adverbs read wrong in a manner slot ("smiled really at the room")
    "almost", "barely", "enough", "especially", "exactly", "hardly", "just",
    "merely", "nearly", "only", "quite", "rather", "really", "simply", "still",
    "too", "truly", "very", "yet", "already", "always", "never", "often",
    "sometimes", "usually", "generally",
    # direction and degree adverbs that read wrong in a manner slot
    "far", "forward", "hardest", "maybe", "possibly", "short", "multiply",
    # tone: aggression, violence and distress do not belong in a bedtime story
    "attack", "brawl", "chase", "drown", "glare", "gloat", "harass", "hurt",
    "jeer", "kidnap", "kill", "mock", "perish", "poach", "quarrel", "scold",
    "scoff", "scorn", "scuffle", "slander", "slobber", "smack", "smash",
    "smirk", "sneer", "snarl", "spank", "spite", "squabble", "stab", "starve",
    "strangle", "suffer", "thrash", "threat", "threaten", "whip",
}

DETAIL_FRAMES = {
    # concrete nouns: object or setting position
    "object": [
        "{name} set the {word} down carefully and stepped back to look.",
        "the {word} stayed right where {name} had left it, quiet and safe.",
        "{friend} peeked at the {word} and let out a happy little gasp.",
        "{name} carried the {word} over to the {place} and settled it in place.",
        "a soft light fell across the {word} near the edge of the {place}.",
        "they found a good spot for the {word} and smoothed everything down.",
        "{friend} held the {word} for a moment, then tucked it in close.",
        "the {place} felt warmer with the {word} sitting right there.",
    ],
    # abstract nouns: reflective, sleepy frames
    "abstract": [
        "{name} thought about {word} for a quiet moment, then smiled.",
        "a warm feeling of {word} spread slowly through the room.",
        "{friend} talked about {word} in a soft, sleepy voice.",
        "they took a slow breath and let {word} fill the quiet space.",
        "{name} whispered about {word} before the lights went out.",
        "the night felt full of {word}, and that was a good thing.",
    ],
    # adjectives and comparatives
    "quality": [
        "everything in the {place} felt {word} and calm.",
        "{name} felt {word} and warm, right down to their toes.",
        "the room looked {word} in the soft evening light.",
        "the quilt felt {word} against {name}'s sleepy cheek.",
        "everything stayed {word} until the very last yawn.",
    ],
    # intransitive verbs in the past tense: motion or state, both read fine
    "action_past_intransitive": [
        "after a while, {name} {word} again, slow and easy.",
        "{friend} {word} nearby, keeping {name} company.",
        "the room went still when {name} {word} for a moment.",
        "in the quiet of the {place}, {friend} {word} and waited.",
    ],
    # transitive verbs in the past tense: the object is the story's own prop
    "action_past_transitive": [
        "{name} {word} {prop} and set it down again.",
        "{friend} {word} {prop} with great care.",
        "they {word} everything until the room felt just right.",
        "{name} {word} it into place before the story began.",
    ],
    # intransitive verbs in their base form: infinitive and modal contexts
    "action_base_intransitive": [
        "it was a good night to {word} and let the day go.",
        "{friend} wanted to {word} for a while before bed.",
        "there was plenty of time to {word} in the sleepy {place}.",
    ],
    # -ing forms: gerunds and participles
    "action_ing": [
        "{name} kept {word} until the whole room felt calm.",
        "there was {name}, still {word} softly in the quiet.",
        "{friend} went on {word} while the evening settled down.",
    ],
    # transitive verbs in their base form
    "action_base_transitive": [
        "there was still time to {word} {prop} before bed.",
        "{name} wanted to {word} {prop} one more time.",
        "maybe they could {word} it together in the morning.",
        "{friend} knew just how to {word} a tricky thing like that.",
    ],
    # -ly adverbs
    "manner": [
        "{friend} nodded {word} and let out a slow breath.",
        "{name} yawned {word} and snuggled down deeper.",
        "the curtains moved {word} in the night breeze.",
        "they whispered {word} so they would not wake anyone.",
        "{name} smiled {word} at the sleepy room.",
    ],
}

_AUTHORED_BANKS = (
    "OPENINGS", "SETTLE_BEATS", "SKY_BEATS", "CHARACTER_PROBLEMS", "HELPER_BEATS",
    "TRY_BEATS", "WARM_BEATS", "QUIET_BEATS", "SLEEP_BEATS", "CLOSINGS",
)


def _strings(obj):
    """Every string inside a nested list/dict/tuple structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _strings(item)


def _authored_bank_words() -> Set[str]:
    """Every word the hand-written banks can put into a story."""
    text = []
    for name in _AUTHORED_BANKS:
        text.extend(_strings(globals()[name]))
    return set(WORD_RE.findall(" ".join(text).lower()))


def _synsets_for(word: str):
    """WordNet entries whose lemma is the lowercase form itself (no proper nouns)."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return []
    return [s for s in wn.synsets(word) if any(l.name() == word for l in s.lemmas())]


def _lemma_candidates(word: str):
    try:
        from nltk.stem import WordNetLemmatizer
    except Exception:
        return [word]
    lem = WordNetLemmatizer()
    return [lem.lemmatize(word, pos=p) for p in ("v", "n", "a", "r")]


def _wordnet_class(word: str):
    """(class, is_abstract) from WordNet, or None when it has no lowercase entry.

    Proper-noun-only synsets are ignored ("Sue" must not make "sue" a noun),
    inflected forms are lemmatised first ("admired" -> "admire"), and the class
    with the most senses wins so that verb-looking words stay verbs.
    """
    synsets = _synsets_for(word)
    if not synsets:
        for lemma in _lemma_candidates(word):
            if lemma != word:
                synsets = _synsets_for(lemma)
                if synsets:
                    break
    if not synsets:
        return None
    counts = {}
    for s in synsets:
        counts[s.pos()] = counts.get(s.pos(), 0) + 1
    if word.endswith("ly") and counts.get("r"):
        return ("manner", False)
    order = ["n", "v", "a", "s", "r"]
    best = max(order, key=lambda p: counts.get(p, 0))
    if best == "n":
        abstract = True
        for s in synsets:
            if s.pos() != "n":
                continue
            for hyper in s.hypernyms()[:1] or []:
                name = hyper.name()
                if "abstraction" not in name and "psychological_feature" not in name:
                    abstract = False
        return ("abstract" if abstract else "object", abstract)
    if best == "v":
        form = _verb_form(word)
        transitive = word not in INTRANSITIVE_VERBS
        if form == "ing":
            return ("action_ing", False)
        if form == "past":
            return ("action_past_transitive" if transitive else "action_past_intransitive", False)
        return ("action_base_transitive" if transitive else "action_base_intransitive", False)
    if best in ("a", "s"):
        return ("quality", False)
    return ("manner", False)


def _verb_form(word: str) -> str:
    """'base' for write-form verbs, 'past' for inflected ones (walked, crept)."""
    try:
        from nltk.stem import WordNetLemmatizer
        lemma = WordNetLemmatizer().lemmatize(word, pos="v")
        if lemma == word:
            return "base"
        return "ing" if word.endswith("ing") else "past"
    except Exception:
        if word.endswith("ing"):
            return "ing"
        return "past" if word.endswith("ed") else "base"


def _is_gerund(word: str) -> bool:
    """True when an -ing word's stem is a known verb (creeping), not a noun (anything)."""
    stem = word[:-3]
    if not stem:
        return False
    if stem.endswith(("at", "it", "av", "ov", "us", "ad", "ed", "id", "ud")):
        stem += "e"
    return bool(_synsets_for(stem)) and any(s.pos() == "v" for s in _synsets_for(stem))


def _heuristic_class(word: str):
    if word.endswith("ly"):
        return "manner"
    if word.endswith(("ous", "ful", "ive", "able", "ible", "ish", "less", "some",
                      "est", "ern", "ic", "al", "y")):
        return "quality"
    past = word.endswith("ed")
    transitive = word not in INTRANSITIVE_VERBS
    if past:
        return "action_past_transitive" if transitive else "action_past_intransitive"
    if word.endswith("ing"):
        if _is_gerund(word):
            return "action_ing"
        return "object"
    if word.endswith("s") and not word.endswith("ss"):
        return "object"
    return "object"


def build_detail_pools(vocabulary_path: str) -> Dict[str, List[str]]:
    """The table's words that the authored banks never use, sorted by class."""
    if not os.path.exists(vocabulary_path):
        return {}
    with open(vocabulary_path) as handle:
        tokens = json.load(handle).get("tokens", [])
    surplus = [
        t for t in tokens
        if not (t.startswith("<") and t.endswith(">"))
        and t not in NOT_USABLE
        and t not in _authored_bank_words()
    ]
    pools: Dict[str, List[str]] = {}
    for word in surplus:
        kind = _wordnet_class(word) or _heuristic_class(word)
        kind = kind[0] if isinstance(kind, tuple) else kind
        pools.setdefault(kind, []).append(word)
    for words in pools.values():
        words.sort()
    return pools


DETAIL_CLASS_ORDER = ["object", "abstract", "quality", "action_ing",
                      "action_past_intransitive", "action_past_transitive",
                      "action_base_intransitive", "action_base_transitive",
                      "manner"]
DETAILS_PER_STORY = 6

_DETAIL_POOLS_CACHE: Dict[str, List[str]] = {}


def detail_pools() -> Dict[str, List[str]]:
    """Cached pools, read from the token table this repository already ships."""
    if not _DETAIL_POOLS_CACHE:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _DETAIL_POOLS_CACHE.update(build_detail_pools(os.path.join(root, "data/vocab.json")))
    return _DETAIL_POOLS_CACHE


def assemble_story(world_id: str, mode: str, place: str, theme: str, variant: int, series_map: dict) -> Tuple[List[str], str]:
    """Generates a complete story adhering to the 200-250 word budget and sleep spine."""
    cfg = series_map[world_id]
    name = cfg["story_name"]
    friends = cfg["friends"]
    props = cfg["props"]

    rng = random.Random(f"{world_id}:{mode}:{place}:{theme}:{variant}")

    friend = friends[variant % len(friends)]
    prop = props[(variant * 3 + 1) % len(props)]

    slots = {
        "name": name,
        "friend": friend,
        "place": place,
        "prop": prop,
    }

    # 1. Opening (1)
    open_pool = OPENINGS.get((world_id, mode), OPENINGS[("bluey", "bedtime")])
    opening_line = open_pool[variant % len(open_pool)].format(**slots)

    # 2. Settle beats (pick 3)
    settle_picked = [b.format(**slots) for b in rng.sample(SETTLE_BEATS, 3)]

    # 3. Sky beats (pick 2)
    sky_picked = [s.format(**slots) for s in rng.sample(SKY_BEATS, 2)]

    # 4. Character Problem (pick 2)
    prob_pool = CHARACTER_PROBLEMS.get(world_id, {}).get(theme, [
        "{friend} was feeling a bit restless and needed some gentle company.",
        "A small question had arisen that needed a patient answer."
    ])
    prob_picked = [p.format(**slots) for p in rng.sample(prob_pool, min(2, len(prob_pool)))]
    if len(prob_picked) < 2:
        prob_picked.append(f"{name} wanted to make sure {friend} felt completely safe.")

    # 5. Helper (pick 1)
    helper_picked = [HELPER_BEATS[rng.randrange(len(HELPER_BEATS))].format(**slots)]

    # 6. Try beats (4 lines)
    try_pool = TRY_BEATS.get(theme, TRY_BEATS["help"])
    try_picked = [t.format(**slots) for t in try_pool]

    # 7. Warm beats (3 lines)
    warm_pool = WARM_BEATS.get(world_id, WARM_BEATS["bluey"])
    warm_picked = [w.format(**slots) for w in warm_pool]

    # 8. Quiet beats (3 lines)
    quiet_picked = [q.format(**slots) for q in rng.sample(QUIET_BEATS, 3)]

    # 8b. Detail beats: spend the words the authored banks never use, one class
    #     at a time, without repeating a class inside a story.
    detail_picked = []
    pools = detail_pools()
    classes = [c for c in DETAIL_CLASS_ORDER if pools.get(c)]
    if classes:
        picked_classes = rng.sample(classes, min(DETAILS_PER_STORY, len(classes)))
        for cls in picked_classes:
            word = rng.choice(pools[cls])
            frame = rng.choice(DETAIL_FRAMES[cls])
            line = frame.format(word=word, **slots)
            detail_picked.append(line[0].upper() + line[1:])

    # 9. Sleep beats (2 lines)
    sleep_picked = [s.format(**slots) for s in SLEEP_BEATS]

    # 10. Closing (1 line)
    close_template = CLOSINGS.get((world_id, mode), CLOSINGS[("bluey", "bedtime")])
    closing_line = close_template.format(**slots)

    all_lines = (
        [opening_line] +
        settle_picked +
        sky_picked +
        prob_picked +
        helper_picked +
        try_picked +
        warm_picked +
        detail_picked +
        quiet_picked +
        sleep_picked +
        [closing_line]
    )

    # Enforce word limits per line (<= 16)
    all_lines = [" ".join(line.split()[:16]) for line in all_lines]

    # Word budget adjustment: target between 205 and 245 words
    def cur_words():
        return sum(len(l.split()) for l in all_lines)

    # If over 248 words, drop one quiet line
    while cur_words() > 248 and len(all_lines) > 18:
        # remove a quiet beat around index -4
        all_lines.pop(-4)

    # If under 205 words, add detail beats (vocabulary-bearing) then settle beats
    extra_settle = [
        f"{name} tucked the edges in snugly and took a slow breath.",
        f"The tranquil night hummed a soothing lullaby across the room.",
        f"{friend} rested peacefully while soft moonlight filled the space."
    ]
    extra_idx = 0
    filler = []
    pools_for_filler = detail_pools()
    filler_classes = [c for c in DETAIL_CLASS_ORDER if pools_for_filler.get(c)]
    for _ in range(6):
        if not filler_classes:
            break
        cls = filler_classes[extra_idx % len(filler_classes)]
        line = rng.choice(DETAIL_FRAMES[cls]).format(
            word=rng.choice(pools_for_filler[cls]), **slots)
        filler.append(line[0].upper() + line[1:])
        extra_idx += 1
    extra_idx = 0
    while cur_words() < 205 and len(all_lines) < 28:
        if extra_idx < len(filler):
            insert_line = filler[extra_idx]
        else:
            insert_line = extra_settle[(extra_idx - len(filler)) % len(extra_settle)]
        extra_idx += 1
        all_lines.insert(len(all_lines) - 2, insert_line)

    char_tag = cfg["token"]
    prompt = f"{char_tag}<MODE={mode.upper()}><PLACE={place.upper()}><THEME={theme.upper()}>"

    return all_lines, prompt


def build_corpus_and_vocab(variants_per_combo: int = 2, target_vocab_size: int = 3530) -> Tuple[List[dict], dict]:
    """Builds the balanced corpus and derives the full closed vocabulary."""
    series_map = {entry["id"]: entry for entry in SERIES_CONFIG}

    stories = []
    all_story_words: Set[str] = set()

    for entry in SERIES_CONFIG:
        world_id = entry["id"]
        for mode in entry["modes"]:
            for place in entry["places"]:
                for theme in THEMES:
                    for v in range(variants_per_combo):
                        lines, prompt = assemble_story(world_id, mode, place, theme, v, series_map)
                        text = prompt + "\n" + "".join(f"<LINE>{l}</LINE>\n" for l in lines) + "<END>\n"

                        row = {
                            "series": world_id,
                            "mode": mode,
                            "place": place,
                            "theme": theme,
                            "variant": v,
                            "prompt": prompt,
                            "lines": lines,
                            "text": text,
                        }
                        stories.append(row)

                        for line in lines:
                            for w in normalize_words(line):
                                all_story_words.add(w)

    print(f"Generated {len(stories)} stories.")
    print(f"Distinct words in generated stories: {len(all_story_words)}")

    # Special tokens
    special_tokens = [
        "<PAD>", "<BOS>", "<EOS>", "<END>", "<LINE>", "</LINE>",
        "<CHAR=BLUEY>", "<CHAR=GABBY>", "<CHAR=ONE>", "<CHAR=AMELIA>", "<CHAR=JUNIE>",
    ]
    for mode in MODES:
        special_tokens.append(f"<MODE={mode.upper()}>")

    all_places = set()
    for entry in SERIES_CONFIG:
        for p in entry["places"]:
            all_places.add(p)
    for p in sorted(all_places):
        special_tokens.append(f"<PLACE={p.upper()}>")

    for t in THEMES:
        special_tokens.append(f"<THEME={t.upper()}>")

    # Load existing vocab.json words so we never drop any existing word
    existing_words = set()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(root_dir, "data/vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            v_data = json.load(f)
            for tok in v_data.get("tokens", []):
                if not (tok.startswith("<") and tok.endswith(">")):
                    existing_words.add(tok)

    combined_words = set(all_story_words) | set(existing_words)
    print(f"Combined words (corpus + existing vocab): {len(combined_words)}")

    # Curate vocabulary to exactly target_vocab_size words
    # Children's storybook words list
    vocab_bank_path = os.path.join(root_dir, "storylm/vocab_bank.txt")
    extra_words = []
    if os.path.exists(vocab_bank_path):
        with open(vocab_bank_path) as f:
            for line in f:
                for w in normalize_words(line):
                    if len(w) > 1:
                        extra_words.append(w)
    else:
        # Load from internal rich word list
        pass

    for w in extra_words:
        combined_words.add(w)

    # If still below target_vocab_size, add common English / children's book words
    sorted_words = sorted(list(combined_words))
    if len(sorted_words) > target_vocab_size:
        # Keep all story words first!
        story_word_set = set(all_story_words)
        essential_words = list(story_word_set)
        remaining_slots = target_vocab_size - len(essential_words)
        if remaining_slots > 0:
            other_words = sorted(list(combined_words - story_word_set))[:remaining_slots]
            sorted_words = sorted(essential_words + other_words)
        else:
            sorted_words = sorted(list(story_word_set))

    tokens = special_tokens + sorted_words
    vocab_data = {
        "count": len(tokens),
        "special_count": len(special_tokens),
        "word_count": len(sorted_words),
        "tokens": tokens,
    }
    used_words = [w for w in sorted_words if w in all_story_words]
    print(f"Vocabulary coverage: {len(used_words)}/{len(sorted_words)} words "
          f"({100 * len(used_words) / max(len(sorted_words), 1):.1f} %) appear in stories; "
          f"{len(sorted_words) - len(used_words)} never used")

    # Strict check: 0 out-of-vocab tokens in any story
    word_set = set(sorted_words)
    missing = set()
    for s in stories:
        for line in s["lines"]:
            for w in normalize_words(line):
                if w not in word_set:
                    missing.add(w)

    if missing:
        print(f"Adding {len(missing)} missing story words into vocab...")
        for m in sorted(list(missing)):
            sorted_words.append(m)
            word_set.add(m)
        sorted_words.sort()
        tokens = special_tokens + sorted_words
        vocab_data = {
            "count": len(tokens),
            "special_count": len(special_tokens),
            "word_count": len(sorted_words),
            "tokens": tokens,
        }

    return stories, vocab_data


def main():
    parser = argparse.ArgumentParser(description="Generate StoryLM 3.5K vocab and training corpus")
    parser.add_argument("--variants", type=int, default=23, help="Variants per combination (default: 23 -> 57,960 stories)")
    parser.add_argument("--target-vocab", type=int, default=3530, help="Target word count (default: 3530)")
    parser.add_argument("--out-corpus", default="data/processed/train.jsonl", help="Output corpus jsonl path")
    parser.add_argument("--out-vocab", default="data/vocab.json", help="Output vocab json path")
    parser.add_argument("--out-series", default="data/series.json", help="Output series json path")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(root, args.out_corpus)
    vocab_path = os.path.join(root, args.out_vocab)
    series_path = os.path.join(root, args.out_series)

    print("=== StoryLM 3.5K Corpus & Vocab Builder ===")
    stories, vocab_data = build_corpus_and_vocab(args.variants, args.target_vocab)

    print(f"\nWriting {len(stories)} stories to {corpus_path}...")
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, "w") as f:
        for s in stories:
            f.write(json.dumps(s) + "\n")

    big_corpus_path = os.path.join(root, "data/processed/train_big.jsonl")
    if corpus_path != big_corpus_path:
        print(f"Writing {len(stories)} stories to {big_corpus_path}...")
        with open(big_corpus_path, "w") as f:
            for s in stories:
                f.write(json.dumps(s) + "\n")

    print(f"Writing vocab table ({vocab_data['count']} tokens, {vocab_data['word_count']} words) to {vocab_path}...")
    with open(vocab_path, "w") as f:
        json.dump(vocab_data, f)

    print(f"Writing updated series profiles to {series_path}...")
    with open(series_path, "w") as f:
        json.dump(SERIES_CONFIG, f, indent=2)

    print("\nSummary Statistics:")
    line_counts = [len(s["lines"]) for s in stories]
    word_counts = [sum(len(l.split()) for l in s["lines"]) for s in stories]
    print(f"Stories:        {len(stories)}")
    print(f"Tokens total:   {vocab_data['count']}")
    print(f"Special tokens: {vocab_data['special_count']}")
    print(f"Words:          {vocab_data['word_count']}")
    print(f"Lines/story:    min {min(line_counts)}, median {int(sum(line_counts)/len(line_counts))}, max {max(line_counts)}")
    print(f"Words/story:    min {min(word_counts)}, median {int(sum(word_counts)/len(word_counts))}, max {max(word_counts)}")

    # Budget validation assertions
    assert min(word_counts) >= 200, f"Violation: story with {min(word_counts)} words (< 200)"
    assert max(word_counts) <= 250, f"Violation: story with {max(word_counts)} words (> 250)"
    assert min(line_counts) >= 13, f"Violation: story with {min(line_counts)} lines (< 13)"
    assert max(line_counts) <= 30, f"Violation: story with {max(line_counts)} lines (> 30)"
    print("ALL VALIDATION CHECKS PASSED: 100% of stories are within 200-250 words, 13-30 lines, <= 16 words/line.")


if __name__ == "__main__":
    main()
