"""HotpotQA dataset for multi-hop question answering optimization.

This module provides a curated subset of 50 real HotpotQA questions (Yang et al., 2018)
for demonstrations and regression tests without requiring external downloads.

Questions are sourced from the official HotpotQA train split (distractor setting),
balanced across difficulty levels (easy/medium/hard) and reasoning types (bridge/comparison).

To use the full HotpotQA benchmark, set the environment variable:
    HOTPOTQA_DATASET_PATH=/path/to/hotpot_dev_distractor_v1.json

The official dataset can be downloaded from:
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

from traigent.evaluators.base import Dataset, EvaluationExample

__all__ = ["dataset_path", "load_case_study_dataset"]

# Environment variable to override with full HotpotQA benchmark
HOTPOTQA_DATASET_PATH_ENV = "HOTPOTQA_DATASET_PATH"


@dataclass(frozen=True)
class _RawExample:
    """Internal representation of a HotpotQA example."""

    question: str
    answer: str
    context: list[str]
    metadata: dict[str, Any]


# 50 real HotpotQA questions from the official train split (distractor setting).
# Distribution: 32 bridge + 18 comparison, 12 easy + 22 medium + 16 hard.
# Source: Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-Hop
# Question Answering", EMNLP 2018.
_RAW_EXAMPLES: Final[list[_RawExample]] = [
    _RawExample(
        question="Rajindar Nath Rehbar is the writer of the song performed by the Ghazal singer married to whom in the 1970s and '80s?",
        answer="Chitra Singh",
        context=[
            "Chitra Singh: Chitra Singh (born Shome) is a well known Indian ghazal singer.",
            "Subhodeep Mukherjee: Subhodeep Mukherjee (Bengali: \u09b6\u09c1\u09ad\u09a6\u09c0\u09aa \u09ae\u09c1\u0996\u09be\u09b0\u099c\u09c0) is an Ghazal singer.",
            "Rajindar Nath Rehbar: Rajindar Nath Rehbar (Urdu:\u0631\u0627\u062c\u0646\u062f\u0631 \u0646\u0627\u062a\u06be \u0631\u06c1\u0628\u0631)(Hindi:\u0930\u093e\u091c\u0947\u0902\u0926\u0930 \u0928\u093e\u0925 \u0930\u0939\u092c\u0930)(born in 5 November 1931) is an Urdu Poet and Bollywood lyricist.",
            "Shanti Hiranand: Shanti Hiranand is an Indian vocalist, classical musician and writer, known for her proficiency as a ghazal singer.",
            "Khalil Haider: Khalil Haider (Urdu: \u200e ) is a Pakistani ghazal singer.",
            "Jagjit Singh: Jagjit Singh, born Jagmohan Singh Dhiman (8 February 1941 \u2013 10 October 2011), was an iconic Indian Ghazal singer, composer and musician.",
            "Pankaj Udhas: Pankaj Udhas (Gujarati: \u0aaa\u0a95\u0a82\u0a9c \u0a89\u0aa7\u0abe\u0ab8 ) is a ghazal singer, hailing from Gujarat in India.",
            "Jasvinder Singh: Jaswinder Singh is a New Age Ghazal Singer from India.",
            "Sithara (singer): Sithara Krishnakumar (born 1 July 1986) is an Indian singer.",
            "Jaspreet 'Jazim' Sharma: Jaspreet 'Jazim' Sharma (born September 14, 1990 in Bathinda, Punjab, India) is an Indian ghazal singer.",
        ],
        metadata={
            "id": "5abecca35542997719eab5ca",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Rajindar Nath Rehbar", 1], ["Jagjit Singh", 1]],
        },
    ),
    _RawExample(
        question="The End of Time is a two-part story and the last story for which lead actor?",
        answer="David Tennant",
        context=[
            "David Tennant: David Tennant (born David John McDonald; 18 April 1971) is a Scottish actor and voice actor.",
            "The Parting of the Ways: \"The Parting of the Ways\" is the thirteenth episode of the revived first series of the British science fiction television programme \"Doctor Who\", which was first broadcast on 18 June 2005.",
            "The End of Time (Doctor Who): The End of Time is a two-part story from the British science fiction television programme \"Doctor Who\".",
            "The Last Sontaran: The Last Sontaran is the first story of Series 2 of \"The Sarah Jane Adventures\" and is a continuation of events from \"Doctor Who\" two-part story \"The Sontaran Strategem\" and \"The Poison Sky\" .",
            "Paradise Island Lost (comics): \"Paradise Island Lost\" is the name to two-part story arc written by Phil Jimenez who also did the artwork, featured in \"Wonder Woman (Vol.",
            "Doomsday (Doctor Who): \"Doomsday\" is the thirteenth and final episode in the second series of the revival of the British science fiction television programme \"Doctor Who\".",
            "Death of the Doctor: Death of the Doctor is a two-part story of \"The Sarah Jane Adventures\" which was broadcast on CBBC on 25 and 26 October 2010.",
            "The Last Story: The Last Story (Japanese: \u30e9\u30b9\u30c8\u30b9\u30c8\u30fc\u30ea\u30fc , Hepburn: Rasuto Sut\u014dr\u012b ) is a Japanese action role-playing game, developed by Mistwalker and AQ Interactive for the Wii video game console.",
            "Thak man-eater: The Thak man-eater was a female Bengal tiger who killed and ate four human victims (two women, two men) between September and November 1938.",
            "The Time of Angels: \"The Time of Angels\" is the fourth episode in the fifth series of British science fiction television series \"Doctor Who\", first broadcast on 24 April 2010 on BBC One.",
        ],
        metadata={
            "id": "5ae0c55e55429945ae959449",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["The End of Time (Doctor Who)", 2], ["David Tennant", 0]],
        },
    ),
    _RawExample(
        question="Is Ween or Superdrag from further north?",
        answer="Ween",
        context=[
            "Superdrag: Superdrag was an American alternative rock band from Knoxville, Tennessee, United States.",
            "Ween: Ween is an American alternative rock band formed in New Hope, Pennsylvania, in 1984 by childhood friends Aaron Freeman and Mickey Melchiondo, better known by their respective stage names, Gene Ween and Dean Ween.",
            "Ween discography: The following is the discography of Ween, a Pennsylvania-based experimental alternative band formed by childhood friends Aaron Freeman and Mickey Melchiondo, better known by their respective stage names, Gene Ween and Dean Ween.",
        ],
        metadata={
            "id": "5a79b7735542994bb9457039",
            "difficulty": "hard",
            "reasoning_type": "comparison",
            "supporting_facts": [["Ween", 0], ["Superdrag", 0]],
        },
    ),
    _RawExample(
        question="Which is marketed primarily to a younger audience, 2600: The Hacker Quarterly or Tiger Beat?",
        answer="Tiger Beat",
        context=[
            "Tiger Beat: Tiger Beat is an American teen fan magazine originally published by The Laufer Company, and marketed primarily to adolescent girls.",
            "Rex Smith: Rex Smith (born September 19, 1955, Jacksonville, Florida) is an American actor and singer.",
            "SmartWool: SmartWool is an American company founded by New England ski instructors Peter and Patty Duke in Steamboat Springs, Colorado in 1994 and acquired by The Timberland Company in 2005.",
            "A Gesheft: A Gesheft (Yiddish: \u05d0 \u05d2\u05e2\u05e9\u05e2\u05e4\u05d8\u200e , \"The Deal\") is a 2005 action film, with a religious message, in the Yiddish language, made by Haredi Jews from Monsey, New York.",
            "OSUNY: OSUNY (Ohio Scientific Users of New York) was a dial-up bulletin board that was run by two different sysops, \"SYSOP\" while in Scarsdale, NY and Frank Roberts in White Plains, NY (914) throughout the 1980s.",
            "PhreakNIC: PhreakNIC is an annual hacker and technology convention held in Nashville, Tennessee.",
            "Special K: Special K is a lightly toasted breakfast cereal manufactured by Kellogg's.",
            "Jim Bray: Jim Bray (born February 23, 1961, Upland, California) is a former competitive artistic roller skater.",
            "Kashmir papier-m\u00e2ch\u00e9: Kashmir papier-m\u00e2ch\u00e9 is a handicraft of Kashmir that was brought by Muslims from Persia in the 15th century.",
            "2600: The Hacker Quarterly: 2600: The Hacker Quarterly is an American seasonal publication of technical information and articles, many of which are written and submitted by the readership, on a variety of subjects including hacking, telephone switching systems, Internet protocols and services, as well as general news concerning the computer \"underground\".",
        ],
        metadata={
            "id": "5a78cac955429974737f7893",
            "difficulty": "medium",
            "reasoning_type": "comparison",
            "supporting_facts": [["2600: The Hacker Quarterly", 0], ["Tiger Beat", 0]],
        },
    ),
    _RawExample(
        question="What's the commercial name for the fictitious branded world inhabited by superheroes and comicbook characters including Doop, Spider-Man and the X-Men?",
        answer="Marvel Universe",
        context=[
            "List of Dinotrux episodes: \"Dinotrux\" is an American computer-animated web television series.",
            "Chakana: The chakana (or Inca Cross) is a stepped cross made up of an equal-armed cross indicating the cardinal points of the compass and a superimposed square.",
            "Dinotrux: Dinotrux is an American computer-animated web television series.",
            "Marvel Universe: The Marvel Universe is the shared universe where the stories in most American comic book titles and other media published by Marvel Entertainment take place.",
            "Grasshopper (comics): The Grasshopper is the name of multiple humorous fictional superheroes appearing in American comic books published by Marvel Comics, all created by Dan Slott.",
            "Marvel 1602: Marvel 1602 is an eight-issue comic book limited series published in 2003 by Marvel Comics.",
            "Doop (comics): Doop is a fictional character appearing in American comic books published by Marvel Comics.",
            "Ultimate Marvel: Ultimate Marvel, later known as Ultimate Comics, was an imprint of comic books published by Marvel Comics, featuring re-imagined and updated versions of the company's superhero characters from the Ultimate Universe.",
            "Fictional universe of Harry Potter: The fictional universe of British author J. K. Rowling's \"Harry Potter\" series of fantasy novels comprises two distinct societies: the Wizarding World and the Muggle world.",
            "Marvel Fairy Tales: Marvel Fairy Tales is a term for three volumes of comic book limited series published by Marvel Comics and written by C. B. Cebulski with art by different artists each issue.",
        ],
        metadata={
            "id": "5abb743b5542996cc5e49ff6",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Doop (comics)", 1], ["Marvel Universe", 0], ["Marvel Universe", 1]],
        },
    ),
    _RawExample(
        question="Novelists Gerhart Hauptmann and Mario Vargas Llosa both won the Nobel Prize in Literature but in which year did Llosa most recently accomplish it?",
        answer="2010",
        context=[
            "Mario Vargas Llosa: Jorge Mario Pedro Vargas Llosa, 1st Marquis of Vargas Llosa (born March 28, 1936), more commonly known as Mario Vargas Llosa ( ; ] ), is a Peruvian writer, politician, journalist, essayist and college professor.",
            "The Perpetual Orgy: The Perpetual Orgy: Flaubert and Madame Bovary (Spanish: \"La org\u00eda perpetua.",
            "The Time of the Hero: The Time of the Hero (original title: \"La ciudad y los perros\", literally \"The City and the Dogs\", 1963) is a 1963 novel by Peruvian writer Mario Vargas Llosa, who won the Nobel Prize in 2010.",
            "Conversation in the Cathedral: Conversation in the Cathedral (original title: \"Conversaci\u00f3n en la catedral\") is a 1969 novel by Peruvian writer and essayist Mario Vargas Llosa, translated by Gregory Rabassa.",
            "Gerhart Hauptmann: Gerhart Johann Robert Hauptmann (15 November 1862 \u2013 6 June 1946) was a German dramatist and novelist.",
            "The Storyteller (Vargas Llosa novel): The Storyteller (Spanish: \"El Hablador\" ) is a novel by Peruvian author and Literature Nobel Prize winner Mario Vargas Llosa.",
            "Peruvian general election, 1990: General elections were held in Peru on 8 April 1990, with a second round of the presidential elections on 10 June.",
            "Jerusalem Prize: The Jerusalem Prize for the Freedom of the Individual in Society is a biennial literary award given to writers whose works have dealt with themes of human freedom in society.",
            "Isaac Humala: Isaac Humala N\u00fa\u00f1ez is a labour lawyer from Ayacucho and the ideological leader of the Movimiento Etnocacerista, a group of ethnic nationalists in Peru.",
            "Marquisate of Vargas Llosa: The Marquisate of Vargas Llosa (Spanish: \"Marquesado de Vargas Llosa\" ) is a hereditary title in the Spanish nobility.",
        ],
        metadata={
            "id": "5a8c6c83554299585d9e3692",
            "difficulty": "easy",
            "reasoning_type": "comparison",
            "supporting_facts": [["Gerhart Hauptmann", 0], ["Mario Vargas Llosa", 0], ["Mario Vargas Llosa", 3]],
        },
    ),
    _RawExample(
        question="Which American dancer teamed with Derek Hough to release the album BHB?",
        answer="Mark Ballas",
        context=[
            "Bethany Joy Lenz discography: This is the discography of Bethany Joy Lenz, an American singer documenting albums, singles and music videos released by Lenz.",
            "Mark Ballas: Mark Alexander Ballas Jr. (born May 24, 1986) is an American dancer, choreographer, singer-songwriter, musician, and actor.",
            "Julianne Hough (album): Julianne Hough is the self-titled debut album of American country singer, and professional dancer, Julianne Hough.",
            "Ballas Hough Band: The Ballas Hough Band was an American pop rock band fronted by Mark Ballas and Derek Hough, both of whom play lead guitar and sing lead vocals.",
            "Derek Hough: Derek Hough ( ; born May 17, 1985) is an American professional Latin and ballroom dancer, choreographer, actor and singer.",
            "The Derek Trucks Band (album): The Derek Trucks Band (often called simply, Derek Trucks) is the debut album by American jam band The Derek Trucks Band, released on October 7, 1997.",
            "Julianne Hough: Julianne Alexandra Hough ( ; born July 20, 1988) is an American dancer, singer, and actress.",
            "World of Dance (TV series): World of Dance is an American reality competition series, hosted by Jenna Dewan Tatum and executive produced by Jennifer Lopez.",
            "BHB (album): BHB is the debut and only album from pop music group Ballas Hough Band formed by Dancing with the Stars professional dancers Derek Hough and Mark Ballas.",
            "Make Your Move (film): Make Your Move (formerly called Cobu 3D, also known as Make Your Move 3D), is a \"Romeo and Juliet\"-inspired 2013 South Korean-American independent dance film starring K-pop singer BoA and ballroom dancer Derek Hough.",
        ],
        metadata={
            "id": "5a75d8335542992db94736ec",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["BHB (album)", 0], ["Mark Ballas", 0]],
        },
    ),
    _RawExample(
        question="Yale University and Rice University, are private research universities, located in which country?",
        answer="United States",
        context=[
            "University of Saskatchewan: The University of Saskatchewan (U of S) is a Canadian public research university, founded in 1907, and located on the east side of the South Saskatchewan River in Saskatoon, Saskatchewan, Canada.",
            "Dermot Moran: Dermot Moran ( ) is an Irish philosopher specialising in phenomenology and in medieval philosophy and also active in the dialogue between analytic and continental philosophy.",
            "International Research Universities Network: The International Research Universities Network (IRUN), initiated in 2006 by Radboud University Nijmegen in the Netherlands, was officially founded during a meeting in September 2007 in Nijmegen.",
            "Emory University: Emory University is a private research university in metropolitan Atlanta, located in the Druid Hills section of DeKalb County, Georgia, United States.",
            "Research university: A research university is a university that expects all its tenured and tenure-track faculty to continuously engage in research, as opposed to merely requiring it as a condition of an initial appointment or tenure.",
            "Woodson Research Center: Woodson Research Center is an archive located in the Fondren Library of Rice University in Houston, Texas.",
            "Yale University: Yale University is an American private Ivy League research university in New Haven, Connecticut.",
            "Hanyang University: Hanyang University is one of the leading private research universities of South Korea, especially in the field of engineering.",
            "Rice University: Rice University, officially William Marsh Rice University, is a private research university located on a 295-acre campus in Houston, Texas, United States.",
            "Jesse H. Jones Graduate School of Business: Located on a 285-acre wooded campus, The Jesse H. Jones Graduate School of Business is the business school of Rice University in Houston, Texas.",
        ],
        metadata={
            "id": "5a76a3905542993735360129",
            "difficulty": "easy",
            "reasoning_type": "comparison",
            "supporting_facts": [["Yale University", 1], ["Rice University", 0]],
        },
    ),
    _RawExample(
        question="The Brown-Forman Corporation owns several brmads throughout the world including which one that is distilled in Shively, Kentucky by the Brown-Forman Corporation?",
        answer="Early Times",
        context=[
            "Old Forester: Old Forester is a brand of Kentucky Straight Bourbon Whiskey produced by the Brown-Forman Corporation.",
            "Jack Daniel's: Jack Daniel's is a brand of Tennessee whiskey and the top selling American whiskey in the world.",
            "Brown-Forman: The Brown-Forman Corporation is one of the largest American-owned companies in the spirits and wine business.",
            "International Alliance for Responsible Drinking: The International Alliance for Responsible Drinking (IARD), headquartered in Washington D.C., is a not-for-profit organization formed to address the global public health issue of harmful drinking.",
            "Heaven Hill: Heaven Hill Distilleries, Inc. is an American, private family-owned and operated distillery company headquartered in Bardstown, Kentucky that produces and markets the Heaven Hill brand of Kentucky Straight Bourbon Whiskey and a variety of other distilled spirits.",
            "Canadian Mist: Canadian Mist is a brand of blended Canadian whisky produced by the Brown-Forman Corporation.",
            "Early Times: Early Times is a brand of Kentucky whiskey distilled in Shively, Kentucky by the Brown-Forman Corporation, one of the largest North American-owned companies in the spirits and wine business.",
            "Brooke Barzun: Brooke Barzun (born Brooke Lee Brown on June 18, 1972) is an art curator and philanthropist based in Louisville, Kentucky with her husband, Matthew Barzun, the former United States Ambassador to the United Kingdom.",
            "Brown College at Monroe Hill: Brown College at Monroe Hill is one of three residential colleges at the University of Virginia.",
            "Owsley Brown Frazier: Owsley Brown Frazier (May 7, 1935 \u2013 August 16, 2012) was a philanthropist from Louisville, Kentucky United States who founded the Frazier History Museum.",
        ],
        metadata={
            "id": "5ae7b7b45542994a481bbdc4",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Brown-Forman", 0], ["Early Times", 0]],
        },
    ),
    _RawExample(
        question="Which plant has more species, Aquilegia or Larrea?",
        answer="Larrea",
        context=[
            "Creosote gall midge: The \"Asphondylia auripila\" group (Diptera: Cecidomyiidae) consists of 15 closely related species of gall-inducing flies which inhabit creosote bush (Zygophyllaceae: \"Larrea tridentata\").",
            "Red columbine: Red columbine can refer to any red-flowered species in the flowering plant genus \"Aquilegia\", especially:",
            "Aquilegia nuragica: Aquilegia nuragica, commonly called Nuragica columbine, is a species of plant in the Ranunculaceae family.",
            "Aquilegia bertolonii: Aquilegia bertolonii, common name Bertoloni columbine, is a species of flowering plant in the family Ranunculaceae, native to Southern France and Italy.",
            "Larrea: Larrea is a genus of flowering plants in the caltrop family, Zygophyllaceae.",
            "Aquilegia barbaricina: Aquilegia barbaricina (also called barbaricina columbine) is a species of plant in the Ranunculaceae family.",
            "Aquilegia grahamii: Aquilegia grahamii is a species of flowering plant in the buttercup family known by the common name Graham's columbine.",
            "Aquilegia: Aquilegia (common names: granny's bonnet or columbine) is a genus of about 60-70 species of perennial plants that are found in meadows, woodlands, and at higher altitudes throughout the Northern Hemisphere, known for the spurred petals of their flowers.",
            "Aquilegia flabellata: Aquilegia flabellata, common name fan columbine or dwarf columbine, is a species of flowering perennial plant in the genus \"Aquilegia\" (columbine), of the family Ranunculaceae.",
            "Aquilegia grata: Aquilegia grata is a species of \"Aquilegia\" native to Serbia, Montenegro and Bosnia.",
        ],
        metadata={
            "id": "5ae48d935542995ad6573d78",
            "difficulty": "hard",
            "reasoning_type": "comparison",
            "supporting_facts": [["Aquilegia", 0], ["Larrea", 0], ["Larrea", 1], ["Larrea", 2]],
        },
    ),
    _RawExample(
        question="Which soft drink, Diet Rite or Julmust, is popular in Sweden during Christmas?",
        answer="Julmust",
        context=[
            "Harry Roberts (inventor): Harry Roberts is the co-inventor of julmust and co-founder of Roberts AB in \u00d6rebro in 1910, Sweden.",
            "Piscola: Piscola or Combinado Nacional (national mix) is a highball cocktail, made of pisco and most commonly a cola drink, that is popular in Chile.",
            "Diet Rite: Diet Rite is a brand of no-calorie soft drinks originally distributed by the RC Cola company.",
            "Patio (soda): Patio Diet Cola was a brand of diet soda introduced by Pepsi in 1963.",
            "Urge (drink): Urge is a citrus flavored soft drink produced by Coca-Cola Norway that was first introduced in the country in 1996, and later on was released in Denmark and Sweden.",
            "Nichols plc: Nichols plc, based in Newton-le-Willows, Merseyside, England, is a company well known for its lead brand Vimto, a fruit flavoured cordial.",
            "Julmust: Julmust (Swedish: \"jul\" \"Yule\" and \"must \" \"not yet fermented juice of fruit or berries\", though there is no such juice in \"julmust\") is a soft drink that is mainly consumed in Sweden around Christmas.",
            "Roberts AB: Roberts AB is the company that makes the syrup for the traditional Swedish drink julmust.",
            "Diet Pepsi: Diet Pepsi and Diet Pepsi Classic Formula Blend (stylized as diet PEPSI CLASSIC SWEETENER BLEND) are no-calorie carbonated cola soft drinks produced by PepsiCo, introduced in 1964 as a variant of Pepsi-Cola with no sugar.",
            "Goombay Punch: Bahamas Goombay Punch is a soft drink that is produced in the Bahamas.",
        ],
        metadata={
            "id": "5a8e504f5542990e94052ac6",
            "difficulty": "medium",
            "reasoning_type": "comparison",
            "supporting_facts": [["Diet Rite", 0], ["Julmust", 0]],
        },
    ),
    _RawExample(
        question="Which band was formed first The Accidentals or Porcupine Tree?",
        answer="Porcupine Tree",
        context=[
            "The Nostalgia Factory: The Nostalgia Factory, subtitled \"...and other tips for amateur golfers\", is the second album to be released by Steven Wilson under the name 'Porcupine Tree'.",
            "Atlanta (album): Atlanta is a download only double-live album by British progressive rock band Porcupine Tree recorded at the Roxy Theatre, Atlanta, United States on 29 October 2007.",
            "Tarquin's Seaweed Farm: Tarquin's Seaweed Farm, subtitled \"Words from a Hessian Sack\", is the first album to be released by Steven Wilson under the name \"Porcupine Tree\".",
            "We Lost the Skyline: We Lost the Skyline (also known as Transmission 7.1) is a live recording by Porcupine Tree, recorded during an in-store performance at Park Avenue CDs in Orlando, Florida, with 200 fans in attendance.",
            "XM (album): XM (also known as Transmission 1.1 and Transmission 1.2) is a live-in-studio album recorded by British band Porcupine Tree in early 2003 as a live album of mostly \"In Absentia\" tracks.",
            "The Accidentals: The Accidentals are an American musical band, formed in Traverse City, Michigan, United States in 2012, by Savannah Buist and Katie Larson.",
            "Anesthetize (Porcupine Tree): Anesthetize is the second live DVD by progressive rock band Porcupine Tree, released on 20 May 2010.",
            "Porcupine Tree: Porcupine Tree were an English rock band formed by musician Steven Wilson in 1987.",
            "Time Flies (song): Time Flies is a single from Porcupine Tree's 2009 studio album \"The Incident\".",
            "XMII: XMII (also known as Transmission 4.1) is a live-in-studio album by British progressive rock band Porcupine Tree, released in June 2005.",
        ],
        metadata={
            "id": "5ab7e5b05542992aa3b8c881",
            "difficulty": "medium",
            "reasoning_type": "comparison",
            "supporting_facts": [["The Accidentals", 0], ["Porcupine Tree", 0]],
        },
    ),
    _RawExample(
        question="Which restaurant chain was founded by Val and Zena Weiler in 1957: Home Run Inn or Valentino's?",
        answer="Valentino's",
        context=[
            "Woodridge, Illinois: Woodridge is a village in DuPage County, Illinois, with portions in Will and Cook counties, and a suburb of Chicago.",
            "Home Run Inn: Home Run Inn is a restaurant chain known for their Chicago-style pizza as well as frozen pizzas.",
            "Doe Run Inn: Doe Run Inn is a restaurant/inn business two miles southeast of Brandenburg, Kentucky.",
            "Babe Ruth's called shot: Babe Ruth's called shot was the home run hit by Babe Ruth of the New York Yankees in the fifth inning of Game 3 of the 1932 World Series, held on October 1, 1932, at Wrigley Field in Chicago.",
            "2017 Major League Baseball Home Run Derby: The 2017 Major League Baseball Home Run Derby was a home run hitting contest between eight batters from Major League Baseball (MLB).",
            "Chicken in the Rough: Chicken in the Rough, also known as Beverly's Chicken in the Rough, is a fried chicken restaurant chain and former franchise.",
            "2014 Major League Baseball Home Run Derby: The 2014 Major League Baseball Home Run Derby (known through sponsorship as the Gillette Home Run Derby) was a home run hitting contest in Major League Baseball (MLB) between five batters each from the American League and National League.",
            "At bats per home run: In baseball statistics, at bats per home run (AB/HR) is a way to measure how frequently a batter hits a home run.",
            "Valentino's: Valentino's is a regional Italian restaurant chain based in Lincoln, Nebraska.",
            "2016 Major League Baseball Home Run Derby: The 2016 Major League Baseball Home Run Derby (known through sponsorship as the T-Mobile Home Run Derby) was a home run hitting contest between eight batters from Major League Baseball (MLB).",
        ],
        metadata={
            "id": "5a73b91455429978a71e9092",
            "difficulty": "medium",
            "reasoning_type": "comparison",
            "supporting_facts": [["Home Run Inn", 0], ["Valentino's", 0], ["Valentino's", 1]],
        },
    ),
    _RawExample(
        question="What is published more frequently, More or Telva.",
        answer="Telva",
        context=[
            "List of fictional places in G.I. Joe: The G.I. Joe: A Real American Hero comic book series was first published by Marvel Comics and later by Devil's Due Productions.",
            "Hydroa vacciniforme: Hydroa vacciniforme (HV) is a very rare, chronic photodermatitis-type skin condition with usual onset in childhood.",
            "MOTi: Timotheus \"Timo\" Romme (born (1987--) 23, 1987 ), better known by his stage name MOTi, is a Dutch electro house DJ and music producer.",
            "The Imp (zine): The Imp is a zine about comics that was written and published by Daniel Raeburn during the late 1990s and early 2000s.",
            "Judah ben Yakar: Judah ben Yakar (d. between 1201 and 1218), talmudist and kabbalist, teacher of Na\u1e25manides.",
            "Gordon Cobbledick: Gordon Cobbledick (December 31, 1898 \u2013 October 2, 1969), was an American sports journalist and author in Cleveland, Ohio.",
            "Roy Croft: Roy Croft is a poet frequently given credit for writing a poem titled \"Love\" and beginning \"I love you not only for what you are, but for what I am when I am with you.\"",
            "More (magazine): More was a women's lifestyle magazine published 10 times a year by the Meredith Corporation with a rate base of 1.3 million and a circulation of 1.8 million.",
            "Telva: Telva is a Spanish language monthly women's magazine published in Madrid, Spain.",
            "Robert Frost: Robert Lee Frost (March26, 1874January29, 1963) was an American poet.",
        ],
        metadata={
            "id": "5a7620b055429976ec32bd25",
            "difficulty": "hard",
            "reasoning_type": "comparison",
            "supporting_facts": [["More (magazine)", 0], ["Telva", 0]],
        },
    ),
    _RawExample(
        question="What type of sports team is Stacey Totman the former head coach of at Texas Tech University?",
        answer="golf",
        context=[
            "Stacey Totman: Stacey Totman is the former head coach of the Texas Tech Red Raiders women's golf team.",
            "Kent Hance: Kent Ronald Hance (born November 14, 1942) is the former Chancellor of the Texas Tech University System.",
            "Texas A&amp;M\u2013Texas Tech football rivalry: The Texas A&M\u2013Texas Tech football rivalry was an American college football rivalry between the Texas A&M Aggies football team of Texas A&M University and Texas Tech Red Raiders football team of Texas Tech University.",
            "Sean Sutton: Sean Patrick Sutton (born October 4, 1968) is an American Basketball Coach and former head coach of the Oklahoma State University men's basketball program from 2006 until April 1, 2008.",
            "Texas Tech Red Raiders football: The Texas Tech Red Raiders football program is a college football team that represents Texas Tech University (variously \"Texas Tech\" or \"TTU\").",
            "1992\u201393 Texas Tech Lady Raiders basketball team: The 1992\u201393 Texas Tech Lady Raiders basketball team represented Texas Tech University in the 1992\u201393 NCAA Division I women's basketball season.",
            "1961\u201362 Texas Tech Red Raiders basketball team: The 1961\u201362 Texas Tech Red Raiders men's basketball team represented Texas Tech University in the Southwest Conference during the 1961\u201362 NCAA University Division men's basketball season.",
            "1960\u201361 Texas Tech Red Raiders basketball team: The 1960\u201361 Texas Tech Red Raiders men's basketball team represented Texas Tech University in the Southwest Conference during the 1960\u201361 NCAA University Division men's basketball season.",
            "Texas Tech Red Raiders golf: The Texas Tech Red Raiders men's and women's golf teams represents Texas Tech University, often referred to as Texas Tech.",
            "2014 Texas Tech Red Raiders baseball team: The 2014 Texas Tech Red Raiders baseball team represents Texas Tech University in the 2014 college baseball season.",
        ],
        metadata={
            "id": "5ab321c6554299233954ff1f",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Stacey Totman", 0], ["Texas Tech Red Raiders golf", 1]],
        },
    ),
    _RawExample(
        question="\"Idle Chatter\" is a popular song adapted from the popular 19th Century Ballet, \"Dance of the Hours\" written by who?",
        answer="Ponchielli",
        context=[
            "Dance of the Hours: Dance of the Hours (Italian: \"Danza delle ore \") is a short ballet and is the act 3 finale of the opera \"La Gioconda\" composed by Amilcare Ponchielli.",
            "Carver Residential Historic District: The Carver Residential Historic District is a national historic district located at Carver, Richmond, Virginia.",
            "Grace Street Commercial Historic District: The Grace Street Commercial Historic District is a national historic district located in Richmond, Virginia.",
            "Carver Industrial Historic District: The Carver Industrial Historic District is a national historic district located at Carver, Richmond, Virginia.",
            "Idle Chatter: \"Idle Chatter\" is a popular song written by Al Sherman and recorded by the Andrews Sisters with the Nelson Riddle Orchestra.",
            "An Ancient Tale (novel): An Ancient Tale.",
            "Thomas Brigham Bishop: Thomas Brigham Bishop (June 29, 1835 - May 15, 1905) (usually referred to as T. Brigham Bishop) is best known as an American composer of popular music.",
            "Orient Historic District: The Orient Historic District is a national historic district in Orient in Suffolk County, New York, United States.",
            "Agrippina Vaganova: Agrippina Yakovlevna Vaganova (Russian: \u0410\u0433\u0440\u0438\u043f\u043f\u0438\u043d\u0430 \u042f\u043a\u043e\u0432\u043b\u0435\u0432\u043d\u0430 \u0412\u0430\u0433\u0430\u043d\u043e\u0432\u0430 ; 26 June 1879 \u2013 5 November 1951) was a Russian ballet teacher who developed the Vaganova method \u2013 the technique which derived from the teaching methods of the old \"Imperial Ballet School\" (today the \"Vaganova Academy of Russian Ballet\") under the \"Premier Ma\u00eetre de Ballet\" Marius Petipa throughout the mid to late 19th century, though mostly throughout the 1880s and 1890s.",
            "Ballet (music): Ballet as a music form progressed from simply a complement to dance, to a concrete compositional form that often had as much value as the dance that went along with it.",
        ],
        metadata={
            "id": "5ae1ec5f5542997283cd22f5",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Idle Chatter", 1], ["Dance of the Hours", 0]],
        },
    ),
    _RawExample(
        question="Careers and Capitol, are which type of entertainment?",
        answer="game",
        context=[
            "Al Coury: Albert Eli \"Al\" Coury (October 21, 1934 \u2013 August 8, 2013) was an Lebanese-American music record executive during the 1970s, vice-president of American record label Capitol Records and co-founder of RSO Records.",
            "Capitol Riverfront: The Capitol Riverfront is a business improvement district (BID) located just south of the United States Capitol between Capitol Hill and the Anacostia River in Washington, D.C. It was created by the District of Columbia City Council and approved by Mayor Fenty in August 2007.",
            "Capitol Records Building: The Capitol Records Building, also known as the Capitol Records Tower, is a Hollywood Boulevard Commercial and Entertainment District building that is located in Hollywood, Los Angeles.",
            "Tonight, I Celebrate My Love: \"Tonight, I Celebrate My Love\" is a song written by Gerry Goffin and Michael Masser recorded as a duet single by Peabo Bryson and Roberta Flack and released in 1983.",
            "Turner's Arena: Turner's Arena was the name given to a 2,000 seat arena, located near the northeast corner of 14th and W Streets, NW in Washington, DC, and originally owned by local wrestling promoter Joe Turner.",
            "Miss Hit and Run: \"Miss Hit and Run\" is a song written by Lynsey de Paul and Barry Blue.",
            "Careers (board game): Careers is a board game first manufactured by Parker Brothers in 1955 for $2.97 US, and was most recently produced by Winning Moves Games.",
            "Capitol (board game): Capitol is a German-style building game set in the ancient Roman Empire, designed by Aaron Weissblum and Alan R. Moon.",
            "Capitol Music Group: Capitol Music Group (abbreviated as CMG) is an American front line umbrella label owned by the Universal Music Group (UMG).",
            "Don Li: Don Li Yat-long is a Hong Kong singer and actor of the Emperor Entertainment Group, Music Icon Records.",
        ],
        metadata={
            "id": "5ae4077355429970de88d87b",
            "difficulty": "easy",
            "reasoning_type": "comparison",
            "supporting_facts": [["Careers (board game)", 0], ["Capitol (board game)", 0]],
        },
    ),
    _RawExample(
        question="Filip Chlap\u00edk played hockey for his team in what stadium?",
        answer="Canadian Tire Centre",
        context=[
            "Ottawa Senators: The Ottawa Senators (French: \"S\u00e9nateurs d'Ottawa\" ) are a professional ice hockey team based in Ottawa, Ontario, Canada.",
            "Robert Hammond: Robert Hammond (born 1981) is an Australian field hockey player from Queensland.",
            "Filip Chlap\u00edk: Filip Chlap\u00edk (born 3 June 1997) is a Czech professional ice hockey player.",
            "Cammi Granato: Catherine Michelle \"Cammi\" Granato (born March 25, 1971) is a retired American female ice hockey player and one of the first women to be inducted into the Hockey Hall of Fame in November 2010.",
            "Brendan Perlini: Brendan Perlini (born April 27, 1996) is an English born Canadian ice hockey forward.",
            "Fred Doherty: Frederick \"Doc\" Doherty (June 15, 1887 \u2013 February 12, 1961) was a Canadian professional ice hockey player.",
            "Reagan Rome: Reagan Rome (born December 29, 1981) is a retired Canadian professional ice hockey defenceman.",
            "Antoine Roussel: Antoine Roussel (born 21 November 1989) is a French/Canadian professional ice hockey left winger currently playing for the Dallas Stars of the National Hockey League (NHL).",
            "International Professional Hockey League: The International Professional Hockey League (IPHL) was the first fully professional Ice hockey league, operating from 1904 to 1907.",
            "Alon Eizenman: Alon Eizenman (born February 9, 1979) is a Canadian and Israeli former ice hockey player.",
        ],
        metadata={
            "id": "5a7eb1135542994959419a53",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Filip Chlap\u00edk", 2], ["Ottawa Senators", 2]],
        },
    ),
    _RawExample(
        question="Who directed the film that starred Val Kilmer's former wife?",
        answer="Wolfgang Petersen",
        context=[
            "Shattered (1991 film): Shattered is a 1991 American neo-noir/psychological thriller starring Tom Berenger, Greta Scacchi, Bob Hoskins, Joanne Whalley and Corbin Bernsen.",
            "The Traveler (2010 film): The Traveler is a 2010 horror film directed by Michael Oblowitz, written by Joseph C. Muscat, and starring Val Kilmer and Dylan Neal.",
            "Joanne Whalley: Joanne Whalley (born 25 August 1964) is an English actress who began her career in 1974.",
            "Top Gun: Top Gun is a 1986 American romantic military action drama film directed by Tony Scott, and produced by Don Simpson and Jerry Bruckheimer, in association with Paramount Pictures.",
            "Heat (1995 film): Heat is a 1995 American crime film written, produced and directed by Michael Mann, and starring Robert De Niro, Al Pacino, and Val Kilmer.",
            "Palo Alto (2013 film): Palo Alto is a 2013 American drama film written and directed by Gia Coppola, based on James Franco's short story collection \"Palo Alto\" (2010).",
            "The Fourth Dimension (film): The Fourth Dimension is a 2012 independent film composed of three segments all created by different directors.",
            "Wonderland (2003 film): Wonderland is a 2003 American crime drama film, co-written and directed by James Cox and based on the real-life Wonderland Murders that occurred in 1981.",
            "MacGruber (film): MacGruber is a 2010 American action comedy film based on the \"Saturday Night Live\" sketch of the same name, itself a parody of action-adventure television series \"MacGyver\".",
            "Batman in film: The fictional superhero Batman, who appears in American comic books published by DC Comics, has appeared in various films since his inception.",
        ],
        metadata={
            "id": "5ab27f22554299340b5254e5",
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["Shattered (1991 film)", 0], ["Shattered (1991 film)", 1], ["Joanne Whalley", 2]],
        },
    ),
    _RawExample(
        question="Which director has won more awards, Oliver Stone or James Cunningham",
        answer="James Cunningham",
        context=[
            "Oliver Stone: William Oliver Stone (born September 15, 1946) is an American screenwriter, film producer, and director of motion pictures and documentaries.",
            "Stanley Weiser: Stanley Weiser is an American screenwriter.",
            "Cunningham automobile: The Cunningham automobile (not connected with the Cunningham Steam Wagon or Briggs Cunningham's cars) was a pioneering American production automobile, one of the earliest vehicles in the advent of the automotive-age produced from 1896 to 1936 in Rochester, New York.",
            "Hoopes-Cunningham Mansion: The Hoopes-Cunningham Mansion is a historic home located at 424 E. Penn St. in Hoopeston, Illinois.",
            "James Cunningham (director): James Cunningham (born 1973) is a New Zealand film director and animator.",
            "James Cunningham, Son and Company: James Cunningham, Son and Company was an American business based in Rochester, New York, initially manufacturing horse-drawn coaches, it eventually went on to develop and produce motorized automobiles from 1908 onward.",
            "Margaret Cunningham: Lady Margaret Cunningham (died in or after 1622) was a Scottish memoirist and correspondent, the daughter of James Cunningham, 7th Earl of Glencairn (1552\u20131630) and his first wife Margaret, daughter of Colin Campbell of Glenorchy.",
            "Claire Simpson: Claire Simpson is a British film editor whose work has been honored with an Academy Award (for Oliver Stone's \"Platoon\") and a BAFTA Film Award for Best Editing for \"The Constant Gardener\".",
            "The Untold History of the United States: The Untold History of the United States (also known as Oliver Stone's Untold History of the United States) is a 2012 documentary series directed, produced, and narrated by Oliver Stone.",
            "Cunninghamia: Cunninghamia is a genus of one or two living species of evergreen coniferous trees in the cypress family Cupressaceae.",
        ],
        metadata={
            "id": "5a7771f75542997042120a46",
            "difficulty": "hard",
            "reasoning_type": "comparison",
            "supporting_facts": [["Oliver Stone", 1], ["Oliver Stone", 2], ["Oliver Stone", 5], ["James Cunningham (director)", 0], ["James Cunningham (director)", 1]],
        },
    ),
    _RawExample(
        question="Black Hawk Down is a 2001 war film based ona book by an author that graduated from what university?",
        answer="Loyola University Maryland",
        context=[
            "Black Hawk War (1865\u201372): The Black Hawk War, or Black Hawk's War, from 1865 to 1872, is the name of the estimated 150 battles, skirmishes, raids, and military engagements between primarily Mormon settlers in Sanpete County, Sevier County and other parts of central and southern Utah, and members of 16 Ute, Paiute, Apache and Navajo tribes, led by a local Ute war chief, Antonga Black Hawk.",
            "Black Hawk War: The Black Hawk War was a brief conflict between the United States and Native Americans led by Black Hawk, a Sauk leader.",
            "Battle of Stillman's Run: The Battle of Stillman's Run, also known as the Battle of Sycamore Creek or the Battle of Old Man's Creek, occurred in Illinois on May 14, 1832.",
            "Black Hawk Down (film): Black Hawk Down is a 2001 war film co-produced and directed by Ridley Scott, from a screenplay by Ken Nolan.",
            "Mark Bowden: Mark Robert Bowden (born July 17, 1951) is an American writer and author.",
            "Battle of Apple River Fort: The Battle of Apple River Fort, occurred on the late afternoon of June 24, 1832 at the Apple River Fort, near present-day Elizabeth, Illinois, when Black Hawk and 200 of his \"British Band\" of Sauk and Fox were surprised by a group of four messengers en route from Galena, Illinois.",
            "Black Hawk Tree: The Black Hawk Tree, or Black Hawk's Tree, was a cottonwood tree located in Prairie du Chien, Wisconsin, United States.",
            "Antonga Black Hawk: Antonga, or Black Hawk (born c. 1830; died September 26, 1870), was a nineteenth-century war chief of the Timpanogos Tribe in what is the present-day state of Utah.",
            "Black Hawk Purchase: The Black Hawk Purchase, which can sometimes be called the Forty-Mile Strip or Scott's Purchase, extended along the West side of the Mississippi River from the north boundary of Missouri North to the Upper Iowa River.",
            "British Band: The British Band was a mixed-nation group of Native Americans commanded by the Sauk leader Black Hawk, which fought against Illinois and Michigan Territory militias during the 1832 Black Hawk War.",
        ],
        metadata={
            "id": "5abd7b3255429924427fcff6",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Black Hawk Down (film)", 1], ["Mark Bowden", 2]],
        },
    ),
    _RawExample(
        question="\"Cook of the House\" is a song written by Paul and Linda McCartney, and released as the B-side to the number 1 single \"Silly Love Songs,\" written in response to who accusing him of writing only \"silly love songs\"?",
        answer="music critics",
        context=[
            "Summer Love Songs: Summer Love Songs is a 2009 compilation of music by the Beach Boys released through Capitol Records.",
            "The Book of Love (The Magnetic Fields song): \"The Book of Love\" is a song written by Stephin Merritt and attributed to The Magnetic Fields, an American indie pop group founded and led by him.",
            "Cook of the House: \"Cook of the House\" is a song written by Paul and Linda McCartney that was first released on Wings' 1976 album \"Wings at the Speed of Sound\".",
            "Love in Song: \"Love in Song\" is a song credited to Paul and Linda McCartney that was released on Wings' 1975 album \"Venus and Mars\".",
            "Silly Love Songs (Glee): \"Silly Love Songs\" is the twelfth episode of the second season of the American musical television series \"Glee\", and the thirty-fourth overall.",
            "Silly Love Songs: \"Silly Love Songs\" is a song written by Paul McCartney and Linda McCartney and performed by Wings.",
            "Songs for a Blue Guitar: Songs for a Blue Guitar is an album by Red House Painters, released on July 22, 1996 in the UK, and a day later in the US.",
            "Wings at the Speed of Sound: Wings at the Speed of Sound is the fifth studio album by Wings, released on 25 March 1976 as a follow-up to their previous album \"Venus and Mars\".",
            "Warm and Beautiful: \"Warm and Beautiful\" is a song credited to Paul and Linda McCartney that was first released by Wings on their 1976 album \"Wings at the Speed of Sound\".",
            "Backwards Traveller/Cuff Link: \"Backwards Traveller\"/\"Cuff Link\" is a medley of two short songs written by Paul McCartney that was first released on Wings' 1978 album \"London Town\".",
        ],
        metadata={
            "id": "5a846f075542992ef85e23f7",
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["Cook of the House", 0], ["Cook of the House", 1], ["Silly Love Songs", 3]],
        },
    ),
    _RawExample(
        question="When was the German fight pilot to which the The Blond Knight of Germany book was dedicated born? ",
        answer="19 April 1922",
        context=[
            "Erich Hartmann: Erich Alfred Hartmann (19 April 1922 \u2013 20 September 1993), nicknamed \"Bubi\" (\"The Kid\") by his German comrades and \"The Black Devil\" by his Soviet adversaries, was a German fighter pilot during World War II and the most successful fighter ace in the history of aerial warfare.",
            "Nesth\u00e4kchen and Her Grandchildren: Else Ury's Nesth\u00e4kchen is a Berlin doctor's daughter, Annemarie Braun, a slim, golden blond, quintessential German girl.",
            "Friedrich Obleser: Friedrich-Erich Obleser (21 February 1923 \u2013 5 June 2004) was a German general in the Bundeswehr.",
            "Heinz R\u00f6kker: Heinz R\u00f6kker (born 20 October 1920) is a former night fighter pilot in the \"Luftwaffe\" of Nazi Germany during World War II.",
            "Kurt Dahlmann: Kurt Dahlmann (born 4 March 1918) is a retired German pilot, attorney, journalist, newspaper editor and political activist.",
            "G\u00fcnther Radusch: G\u00fcnther Radusch (11 November 1912 \u2013 29 July 1988) was a German pilot in the \"Luftwaffe\" pilot during World War II.",
            "The Blond Knight of Germany: The Blond Knight of Germany is a book by the American authors Trevor J. Constable and Raymond F. Toliver dedicated to the life and career of the German fighter pilot of World War II, Erich Hartmann.",
            "Siegfried Barth: Siegfried Barth (23 January 1916 \u2013 19 December 1997) was a German bomber pilot in the Luftwaffe during World War II and commander of the fighter-bomber wing Jagdbombergeschwader 32 (JaBoG 32) of the German Air Force.",
            "Bernhard Jope: Bernhard Jope (10 May 1914 \u2013 31 July 1995) was a German bomber pilot during World War II.",
            "Ulrich Diesing: Ulrich Diesing (12 March 1911 \u2013 17 April 1945) was a German pilot in the Luftwaffe during World War II.",
        ],
        metadata={
            "id": "5ac3226d5542995ef918c113",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["The Blond Knight of Germany", 0], ["Erich Hartmann", 0]],
        },
    ),
    _RawExample(
        question="North is a 1994 American comedy drama adventure film, which American actress made her film debut in the fantasy comedy?",
        answer="Scarlett Johansson",
        context=[
            "Kishori Ballal: Kishori Ballal is a veteran Indian actress who is known for her works in Kannada cinema.",
            "North (1994 film): North is a 1994 American comedy drama adventure film directed by Rob Reiner and starring an ensemble cast including Elijah Wood, Jon Lovitz, Jason Alexander, Alan Arkin, Dan Aykroyd, Kathy Bates, Faith Ford, Graham Greene, Julia Louis-Dreyfus, Reba McEntire, John Ritter, and Abe Vigoda, with cameos by Bruce Willis and a 9-year-old Scarlett Johansson (in her film debut).",
            "Rimpi Das: Rimpi Das is an Indian actress and model, who works in Assamese cinema and Hindi television industry.",
            "Scarlett Johansson: Scarlett Johansson ( ; born November 22, 1984) is an American actress, model, and singer.",
            "The Mask (film): The Mask is a 1994 American superhero fantasy comedy film directed by Charles Russell, produced by Bob Engelman, and written by Mike Werb, based on the comic series of the same name distributed by Dark Horse Comics.",
            "War of the Buttons (1994 film): War of the Buttons is a 1994 Irish-French drama adventure film directed by John Roberts.",
            "Tim Burton: Timothy Walter Burton ( ; born August 25, 1958) is an American film director, producer, artist, writer, and animator.",
            "Duma (2005 film): Duma is a 2005 American family drama adventure film about a young South African boy's friendship with an orphaned cheetah from a story by Carol Flint and Karen Janszen, set in the country of South Africa and based on \"How It Was with Dooms\" by Carol Cawthra Hopcraft and Xan Hopcraft.",
            "Prince of Jutland: Prince of Jutland, also known as Royal Deceit, is a 1994 drama adventure film co-written and directed by Gabriel Axel and starring Christian Bale, Gabriel Byrne and Helen Mirren.",
            "Vazante (film): Vazante is a Brazilian-Portuguese historical period drama adventure film about slavery in 1820s Brazil, directed by Daniela Thomas in her feature film debut.",
        ],
        metadata={
            "id": "5a74dd255542996c70cfae21",
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["North (1994 film)", 0], ["Scarlett Johansson", 0]],
        },
    ),
    _RawExample(
        question="4:21 ...The Day After is an album produced by man who was the leader of what group? ",
        answer="Wu-Tang Clan",
        context=[
            "Some People Change: Some People Change is the fifth studio album by the American country music duo Montgomery Gentry.",
            "Nothing but the Truth (Southern Sons album): Nothing But The Truth is the second studio album by Australian music group Southern Sons.",
            "4:21... The Day After: 4:21 ...The Day After is the fourth studio album by American rapper and Wu-Tang Clan member Method Man.",
            "Mistaken Identity (Delta Goodrem album): Mistaken Identity is the second album by Australian singer Delta Goodrem, released in Australia on 8 November 2004, a day before Goodrem's twentieth birthday, by Epic and Daylight Records.",
            "We All Get Lucky Sometimes: We All Get Lucky Sometimes is the fourth studio album by American country music singer Lee Roy Parnell.",
            "Bethany Joy Lenz discography: This is the discography of Bethany Joy Lenz, an American singer documenting albums, singles and music videos released by Lenz.",
            "Steal Another Day: Steal Another Day is an album released in 2003 by country music artist Steve Wariner and his first studio album for SelecTone Records.",
            "RZA: Robert Fitzgerald Diggs (born July 5, 1969), better known by his stage name RZA ( \"rizza\"), is an American rapper, record producer, musician, actor, filmmaker and author.",
            "Loose, Loud &amp; Crazy: Loose, Loud, & Crazy is the third studio album of American country music singer Kevin Fowler, and his fourth album overall.",
            "The Notorious Cherry Bombs (album): The Notorious Cherry Bombs is the self-titled debut album by the American country music group The Notorious Cherry Bombs, a band that formerly served as country singer Rodney Crowell's backing band in the 1980s.",
        ],
        metadata={
            "id": "5ac562655542993e66e823c6",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["4:21... The Day After", 3], ["RZA", 1]],
        },
    ),
    _RawExample(
        question="What song by the former bass guitarist, Singer and Songwriter with The Beatles, is on his Sixth solo album which was the first album of entirely new music?",
        answer="Stranglehold",
        context=[
            "Indigo Algorithm (Ai no Denshi Kisuuhou): Indigo Algorithm -Ai no Denshi Kisuuhou- (\"Indigo Algorithm\"\uff0d\u85cd\u306e\u96fb\u601d\u57fa\u6570\u6cd5\uff0d), also known as Quantum Mechanics Rainbow II: Indigo Algorithm, is the seventh (sixth of entirely new music) solo album by artist Daisuke Asakura.",
            "Stranglehold (Paul McCartney song): \"Stranglehold\" is a song by Paul McCartney, the former bass guitarist, Singer and Songwriter with The Beatles.",
            "Get Heavy: Get Heavy is the debut album by Finnish hard rock band Lordi, released in 2002.",
            "Violet Meme (Murasaki no Jyouhoudentatsu Chi): Violet Meme -Murasaki no Jyouhoudentatsu Chi-(\"Violet Meme\" \uff0d\u7d2b\u306e\u60c5\u5831\u4f1d\u9054\u5024\uff0d), also known as Quantum Mechanics Rainbow I: Violet Meme, is the sixth (fifth of entirely new music) solo album by artist Daisuke Asakura.",
            "The Rock (John Entwistle album): The Rock is the sixth solo studio album by the English musician John Entwistle, formerly of the Who.",
            "Ordinary Average Guy: Ordinary Average Guy is the ninth solo studio album, and its title-track single (second on the playlist), by American singer-songwriter and multi-instrumentalist Joe Walsh.",
            "Blue Resolution (Ao no Shikaku Kaiseki do): \"Blue Resolution -Ao no Shikaku Kaiseki do - (\"Blue Resolution\"\uff0d\u9752\u306e\u601d\u899a\u89e3\u6790\u5ea6\uff0d), also known as Quantum Mechanics Rainbow III: Blue Resolution, is the eighth (seventh of entirely new music) solo album by artist Daisuke Asakura.",
            "Press to Play: Press to Play is the sixth post-Beatles studio album by the English musician Paul McCartney (outside of Wings' body of work), released in August 1986.",
            "Clou: Clou was a Czech rock band from Prague, Czech Republic, formed in 2001.",
            "Boys and Girls (album): Boys and Girls is the sixth solo studio album by the English singer and songwriter Bryan Ferry, released in June 1985 by E.G. Records.",
        ],
        metadata={
            "id": "5a78dbd4554299029c4b5eb7",
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["Stranglehold (Paul McCartney song)", 0], ["Press to Play", 1]],
        },
    ),
    _RawExample(
        question="What part of a compilation does Shake Your Coconuts and Junior Senior have in common?",
        answer="single",
        context=[
            "Northern Potter Junior Senior High School: Northern Potter Junior Senior High School is a diminutive, rural, public junior senior high school located in Ulysses, Potter County, Pennsylvania.",
            "Shake Your Coconuts: \"Shake Your Coconuts\" is a song by Danish indie pop duo Junior Senior.",
            "Elderton High School: Elderton Junior Senior High School was a tiny, rural, public junior senior high school in Elderton in the U.S. state of Pennsylvania.",
            "Riverside Junior-Senior High School: Riverside Junior Senior High School is small public school located at: 310 Davis Street, Taylor, Lackawanna County, Pennsylvania.",
            "Austin High School (Austin, Pennsylvania): Austin Area Junior Senior High School is a diminutive, public high school in southern and rural Potter County, Pennsylvania.",
            "Cowanesque Valley Junior Senior High School: Cowanesque Valley Junior Senior High School is a diminutive, rural public high school.",
            "Montrose Area Junior Senior High School: Montrose Area Junior Senior High School is located at 50 High School Road, Montrose, Susquehanna County, Pennsylvania.",
            "Junior Senior: Junior Senior was a pop duo from Denmark.",
            "Carbondale Area Junior Senior High School: Carbondale Area junior Senior High School is located at 101 Brooklyn Street, Carbondale.",
            "Canton Junior Senior High School: Canton Junior Senior High School is a diminutive, rural public combined junior senior high school located at 509 E Main Street, Canton, Pennsylvania.",
        ],
        metadata={
            "id": "5a8e2a4e554299068b959e72",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Shake Your Coconuts", 1], ["Junior Senior", 2]],
        },
    ),
    _RawExample(
        question="Ladislao Vajda and Todd Graff, have which mutual occupation?",
        answer="film director",
        context=[
            "Afternoon of the Bulls: Afternoon of the Bulls (Spanish: Tarde de toros ) is a 1956 Spanish drama film directed by Ladislao Vajda.",
            "Ladislaus Vajda: Ladislaus Vajda (born L\u00e1szl\u00f3 Vajda; 18 August 1877 \u2013 10 March 1933) was a Hungarian screenwriter.",
            "Nikka Graff Lanzarone: Nikka Graff Lanzarone (Born November 20, 1983) is an actress and dancer.",
            "Where Is This Lady?: Where Is This Lady?",
            "Ladislao Vajda: Ladislao Vajda (born L\u00e1szl\u00f3 Vajda Weisz; 18 August 1906, Budapest \u2013 25 March 1965, Barcelona) was a Hungarian film director who made films in Spain, Portugal, the United Kingdom, Italy and Germany.",
            "Adventures of the Barber of Seville: Adventures of the Barber of Seville (Spanish: Aventuras del barbero de Sevilla ) is a 1954 Spanish comedy film directed by Ladislao Vajda.",
            "Do\u00f1a Francisquita (film): Do\u00f1a Francisquita is a 1953 Spanish comedy film directed by Ladislao Vajda.",
            "Todd Graff: Todd Graff (born October 22, 1959) is an American actor, writer and director, best known for his 2003 independent film \"Camp\" and his role as Alan \"Hippy\" Carnes in the 1989 science fiction film \"The Abyss\".",
            "The passer-through-walls: The passer-through-walls (French: \"Le Passe-muraille\"), translated as \"The Man Who Walked through Walls\", \"The Walker-through-Walls\" or \"The Man who Could Walk through Walls\", is a short story published by Marcel Aym\u00e9 in 1943.",
            "Call of the Blood: Call of the Blood is a 1948 British-Italian drama film directed by John Clements and Ladislao Vajda and starring Clements, Kay Hammond and John Justin.",
        ],
        metadata={
            "id": "5a8283d355429966c78a6a3d",
            "difficulty": "easy",
            "reasoning_type": "comparison",
            "supporting_facts": [["Ladislao Vajda", 0], ["Todd Graff", 0]],
        },
    ),
    _RawExample(
        question="Which hospital has more beds, Children's National Medical Center or MedStar Washington Hospital Center?",
        answer="MedStar Washington Hospital Center",
        context=[
            "Legacy Holladay Park Medical Center: Legacy Holladay Park Medical Center is a hospital located in Portland, Oregon, United States.",
            "Murphy Medical Center: Murphy Medical Center (MMC) is a hospital located in Murphy, North Carolina certified by the United States Department of Health and Human Services.",
            "Providence Portland Medical Center: Providence Portland Medical Center, located at 4805 NE Glisan St. in the North Tabor neighborhood of Portland, Oregon, is a full-service medical center specializing in cancer and cardiac care.",
            "Children's National Medical Center: Children\u2019s National Medical Center (formerly DC Children\u2019s Hospital) is ranked among the top 10 children\u2019s hospitals in the country by \"U.S. News & World Report.\"",
            "List of hospitals in North Carolina: This is a list of hospitals in North Carolina.",
            "Banner University Medical Center Tucson: Banner - University Medical Center Tucson (BUMCT), formerly University Medical Center and the University of Arizona Medical Center, is a private, non-profit, 487-bed acute-care hospital located on the campus of the University of Arizona in Tucson, Arizona.",
            "Brad Wenstrup: Brad Robert Wenstrup (born June 17, 1958) is an American politician, Army Reserve officer, and Doctor of Podiatric Medicine, who has been the U.S. Representative for Ohio 's 2 congressional district since 2013.",
            "MedStar Washington Hospital Center: MedStar Washington Hospital Center is the largest private hospital in Washington, D.C. A member of MedStar Health, the not-for-profit Hospital Center is licensed for 926 beds.",
            "Vassar Brothers Medical Center: Vassar Brother Medical Center (locally known as Vassar Hospital or VBMC) is a major medical facility located in the city of Poughkeepsie, New York that is a member of the Health Quest network, a nonprofit family of hospitals and healthcare centers in the Hudson Valley area.",
            "MedStar Georgetown University Hospital: MedStar Georgetown University Hospital is one of the national capital area's oldest academic teaching hospitals.",
        ],
        metadata={
            "id": "5a74d9d555429974ef308c7d",
            "difficulty": "medium",
            "reasoning_type": "comparison",
            "supporting_facts": [["Children's National Medical Center", 4], ["MedStar Washington Hospital Center", 0]],
        },
    ),
    _RawExample(
        question="What breed besides the Xoloitzcuintli is specifically bred for alopecia?",
        answer="Chinese Crested Dog",
        context=[
            "Wolf hunting with dogs: Wolf hunting with dogs is a method of wolf hunting which relies on the use of hunting dogs.",
            "Canine follicular dysplasia: Follicular dysplasia is a genetic disease of dogs causing alopecia, or hair loss.",
            "Border Collie: The Border Collie is a working and herding dog breed developed in the Anglo-Scottish border region for herding livestock, especially sheep.",
            "Mexican Hairless Dog: The Xoloitzcuintli ( ; Nahuatl pronunciation: ), or Xolo for short, is a hairless breed of dog, found in toy, miniature, and standard sizes.",
            "Cortexolone 17\u03b1-propionate: Cortexolone 17\u03b1-propionate (developmental code name CB-03-01; tentative brand names Breezula (for acne), Winlevi (for androgenic alopecia)), or 11-deoxycortisol 17\u03b1-propionate, is a synthetic, steroidal antiandrogen \u2013 specifically, an androgen receptor antagonist \u2013 that is under development by Cassiopea and Intrepid Therapeutics for use as a topical medication in the treatment of androgen-dependent conditions including acne vulgaris and androgenic alopecia (male-pattern hair loss).",
            "Black and Tan Virginia Foxhound: The Black and Tan Virginia Foxhound is an American foxhound breed.",
            "Hair prosthesis: A hair prosthesis (or cranial prosthesis) is a custom-made wig specifically designed for patients who have lost their hair as a result of medical conditions or treatments, such as alopecia areata, alopecia totalis, trichotillomania, chemotherapy, or any other clinical disease or treatment resulting in hair loss.",
            "Friesian Sporthorse: The Friesian Sporthorse is a Friesian crossbred of sport horse type.",
            "Beulah Speckled Face: The Beulah Speckled Face is a breed of domestic sheep originating in the United Kingdom.",
            "Alopecia in animals: Alopecia is a disease which can affect other animals besides humans.",
        ],
        metadata={
            "id": "5a875003554299211dda2bdd",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Canine follicular dysplasia", 4], ["Mexican Hairless Dog", 0]],
        },
    ),
    _RawExample(
        question="Which astronaut carried copies of Swedish singer Darin's single into space with him and is the first Swedish citizen in space?",
        answer="Arne Christer Fuglesang",
        context=[
            "Money for Nothing (Darin song): \"Money for Nothing\" is the debut single released by Swedish singer Darin in 2005.",
            "Perfect (Darin song): \"Perfect\" is a song recorded by Swedish singer Darin.",
            "Step Up (Darin song): \"Step Up\" is the third single released by Swedish singer Darin.",
            "Christer Fuglesang: Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.",
            "Breathing Your Love: \"Breathing Your Love\" is a song by the Swedish singer Darin featuring vocals by singer Kat DeLuna and the first single from \"Flashback\".",
            "Break the News: Break the News is the third studio album by the Swedish singer Darin.",
            "Insanity (song): \"Insanity\" is a song written by Peter Mansson, Patric Sarin, Darin Zanyar and recorded by Swedish singer Darin.",
            "Want Ya!: \"Want Ya!\"",
            "You're Out of My Life: \"You're Out of My Life\" is a song recorded by Swedish singer Darin and was Darin's entry in Melodifestivalen 2010.",
            "Desire (Darin song): \"Desire\" is a song recorded by Swedish singer Darin.",
        ],
        metadata={
            "id": "5a7275ce5542994cef4bc2af",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Breathing Your Love", 3], ["Christer Fuglesang", 0], ["Christer Fuglesang", 1]],
        },
    ),
    _RawExample(
        question="Superfast!, is a 2015 American parody comedy film written and directed by Jason Friedberg and Aaron Seltzer, and is a parody of which American franchise based on a series of action films that is largely concerned with illegal street racing and heists, and includes material in various other media that depicts characters and situations from the films?",
        answer="The Fast and the Furious",
        context=[
            "Meet the Spartans: Meet the Spartans is a 2008 American parody film directed by Jason Friedberg and Aaron Seltzer.",
            "The Fast and the Furious: The Fast and the Furious (also known as Fast & Furious) is an American franchise based on a series of action films that is largely concerned with illegal street racing and heists, and includes material in various other media that depicts characters and situations from the films.",
            "Disaster Movie: Disaster Movie is a 2008 American comedy film written and directed by Jason Friedberg and Aaron Seltzer, and stars Matt Lanter, Vanessa Minnillo, Gary \"G Thang\" Johnson, Crista Flanagan, Ike Barinholtz, Carmen Electra, Tony Cox, and Kim Kardashian in her feature film acting debut.",
            "Spy Hard: Spy Hard is a 1996 American spy comedy film parody starring Leslie Nielsen and Nicollette Sheridan, parodying James Bond movies and other action films.",
            "The Starving Games: The Starving Games is a 2013 American parody film based on \"The Hunger Games\" and directed by Jason Friedberg and Aaron Seltzer.",
            "Superfast!: Superfast!",
            "Rick Friedberg: Rick Friedberg is an American film and television director and producer, and the father of Jason Friedberg, one half of a comedy writing duo with Aaron Seltzer (\"Meet The Spartans\", \"Disaster Movie\").",
            "Epic Movie: Epic Movie is a 2007 American comedy film directed and written by Jason Friedberg and Aaron Seltzer and produced by Paul Schiff.",
            "Jason Friedberg and Aaron Seltzer: Jason Friedberg (born October 13, 1971) and Aaron Seltzer (born January 12, 1974) are an American-Canadian film director and screenwriter team known for making parody movies that have received extremely unfavorable reviews, but have done well at the box office.",
            "Best Night Ever: Best Night Ever is a 2013 American found footage comedy film written and directed by Jason Friedberg and Aaron Seltzer and produced by Jason Blum, Friedberg and Seltzer.",
        ],
        metadata={
            "id": "5adfb6e8554299603e418386",
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["Superfast!", 2], ["The Fast and the Furious", 0]],
        },
    ),
    _RawExample(
        question="What group features half soprano singer Virpi Moskari?",
        answer="Rajaton",
        context=[
            "Thomas-McJunkin-Love House: Thomas-McJunkin-Love House is a historic home located at Charleston, West Virginia.",
            "Elgin Schoolhouse State Historic Site: Elgin Schoolhouse State Historic Site is a state park property in Nevada, United States, preserving a historic one-room schoolhouse that operated from 1922 to 1967.",
            "All Is Dream: All Is Dream is the fifth studio album by American rock band Mercury Rev.",
            "Belgrade Marathon: The Belgrade Marathon is a marathon race held annually in Belgrade since 1988.",
            "Nancy Argenta: Nancy Maureen Argenta (born Nancy Maureen Herbison on January 17, 1957) is a Canadian soprano singer, best known for performing music from the pre-classical era.",
            "Manuela Kraller: Manuela Kraller (born 1 August 1981) is a German soprano singer from Ainring.",
            "Zulma Bouffar: Zulma Madeleine Boufflar, known as Zulma Bouffar, (24 May 1841 \u2013 20 January 1909), was a French actress and soprano singer, associated with the op\u00e9ra-bouffe of Paris in the second half of the 19th century who enjoyed a successful career around Europe.",
            "Mezzo-soprano: A mezzo-soprano or mezzo ( , ; ] meaning \"half soprano\") is a type of classical female singing voice whose vocal range lies between the soprano and the contralto voice types.",
            "Keys to Ascension: Keys to Ascension is the fourth live and fifteenth studio album by the English rock band Yes.",
            "Virpi Moskari: Virpi Moskari is a Mezzo-soprano, and a founder member of the Finnish a cappella group, Rajaton.",
        ],
        metadata={
            "id": "5ae447645542995ad6573d27",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Virpi Moskari", 0], ["Mezzo-soprano", 0]],
        },
    ),
    _RawExample(
        question="Which person who secretly recorded people in the ACORN 2009 undercover videos controversy was born in 1984?",
        answer="James Edward O'Keefe III",
        context=[
            "ACORN 2009 undercover videos controversy: In 2009, workers at offices of the Association of Community Organizations for Reform Now (ACORN), a non-profit organization that had been involved for nearly 40 years in voter registration, community organizing and advocacy for low- and moderate-income people, were secretly recorded by conservative activists Hannah Giles and James O'Keefe \u2013 and the videos \"heavily edited\" to create a misleading impression of their activities.",
            "EXCOMM: The Executive Committee of the National Security Council (commonly referred to as simply the Executive Committee or ExComm) was a body of United States government officials that convened to advise President John F. Kennedy during the Cuban Missile Crisis in 1962.",
            "Paul Schockem\u00f6hle: Paul Schockem\u00f6hle (born March 22, 1945, in M\u00fchlen, Oldenburg) is a former German showjumper.",
            "Adrian Schoolcraft: Adrian Schoolcraft (born 1976) is a former New York City Police Department (NYPD) officer who secretly recorded police conversations from 2008 to 2009.",
            "Oleksy tapes: The Oleksy tapes are tapes secretly recorded during a 2006 conversation between J\u00f3zef Oleksy and Aleksander Gudzowaty.",
            "Rodrigue Mugaruka Katembo: Rodrigue Mugaruka Katembo (born 1976 in Katana, South Kivu, DR Congo) is a park ranger from the Democratic Republic of the Congo.",
            "I'll See You in Court: \"I'll See You in Court\" is the tenth episode of the third season from the TV comedy series \"Married... with Children\".",
            "Mahmoud Badr: Mahmoud Badr (Arabic: \u0645\u062d\u0645\u0648\u062f \u0628\u062f\u0631\u200e \u200e ; born 1985) is an Egyptian activist and journalist.",
            "James O'Keefe: James Edward O'Keefe III (born June 28, 1984) is an American conservative political activist.",
            "Planned Parenthood 2015 undercover videos controversy: Planned Parenthood 2015 undercover videos controversy",
        ],
        metadata={
            "id": "5a7d078d55429909bec76904",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["ACORN 2009 undercover videos controversy", 0], ["James O'Keefe", 0]],
        },
    ),
    _RawExample(
        question="Caldera de Taburiente National Park and Teide National Park are located in which island chain?",
        answer="Canary Islands, Spain",
        context=[
            "Echium gentianoides: Echium gentianoides is a synonym of \"Echium thyrsiflorum\" , a flowering plant in the borage family Boraginaceae with brilliant blue tubular flowers.",
            "Teide National Park: Teide National Park (Spanish: \"Parque nacional del Teide\" , ] ) is a national park located in Tenerife (Canary Islands, Spain).",
            "Idafe Rock: Idafe Rock is a natural stone pillar located in Caldera de Taburiente National Park on the island of La Palma in the Canary Islands.",
            "Echium wildpretii subsp. trichosiphon: Echium wildpretii\" subsp.",
            "Adenocarpus viscosus: Adenocarpus viscosus is a shrubby species of flowering plant in the legume family Fabaceae, subfamily Faboideae.",
            "Roque de los Muchachos: Roque de los Muchachos (English: \"Rock of the Boys\") is a rocky mound at the highest point on the island of La Palma in the Canary Islands, Spain.",
            "Barlovento, Santa Cruz de Tenerife: Barlovento (Spanish for windward) is a municipality in the northern part of the island of La Palma, one of the Canary Islands, and a part of the province of Santa Cruz de Tenerife.",
            "Roque Cinchado: The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).",
            "Caldera de Taburiente National Park: Caldera de Taburiente National Park (Spanish: \"Parque Nacional de la Caldera de Taburiente\" ) is large geological feature on the island of La Palma, Canary Islands, Spain.",
            "Valles Caldera: Valles Caldera (or Jemez Caldera) is a 13.7 mi wide volcanic caldera in the Jemez Mountains of northern New Mexico.",
        ],
        metadata={
            "id": "5a7530fa5542993748c897e2",
            "difficulty": "hard",
            "reasoning_type": "comparison",
            "supporting_facts": [["Caldera de Taburiente National Park", 0], ["Teide National Park", 0]],
        },
    ),
    _RawExample(
        question="Where is the head coach of the 1999-2000 Kentucky Wildcats basketball team currently coaching?",
        answer="University of Memphis",
        context=[
            "Davidson Wildcats men's basketball: The Davidson Wildcats basketball team is the basketball team that represents Davidson College in Davidson, North Carolina, in the NCAA Division I.",
            "2007\u201308 Kentucky Wildcats men's basketball team: The 2007\u201308 Kentucky Wildcats men's basketball team represented the University of Kentucky in the college basketball season of 2007\u20132008.",
            "2000 Kentucky Wildcats football team: The 2000 Kentucky Wildcats football team represented the University of Kentucky during the 2000 NCAA Division I-A football season.",
            "New Hampshire Wildcats men's basketball: The New Hampshire Wildcats Basketball team is the basketball team that represent the University of New Hampshire in Durham, New Hampshire.",
            "Yeng Guiao: Joseller \"Yeng\" Guiao (Born March 19, 1959) is a Filipino professional basketball head coach, politician, commentator and sports commissioner.",
            "Alumni Gymnasium (University of Kentucky): Alumni Gymnasium is a building on the University of Kentucky campus in Lexington, Kentucky.",
            "Florida\u2013Kentucky men's basketball rivalry: The Florida\u2013Kentucky men's basketball rivalry is a semi-annual rivalry between the Florida Gators and the Kentucky Wildcats basketball teams.",
            "2002\u201303 Kentucky Wildcats men's basketball team: The 2002\u201303 Kentucky Wildcats men's basketball team represented University of Kentucky.",
            "1999\u20132000 Kentucky Wildcats men's basketball team: The 1999-2000 Kentucky Wildcats men's basketball team represented University of Kentucky in the 1999-2000 NCAA Division I men's basketball season.",
            "Tubby Smith: Orlando Henry \"Tubby\" Smith (born June 30, 1951) is an American college basketball coach.",
        ],
        metadata={
            "id": "5ae3af805542991a06ce9a0f",
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["1999\u20132000 Kentucky Wildcats men's basketball team", 1], ["Tubby Smith", 1]],
        },
    ),
    _RawExample(
        question="Where is the headquarters of the company who launched the Fridge advertising campaign?",
        answer="London, England",
        context=[
            "Where do you want to go today?: \u201cWhere do you want to go today?\u201d",
            "Elf Yourself: Elf Yourself is an American interactive viral website where visitors can upload images of themselves or their friends, see them as dancing elves, and have the option to post the created video to other sites or save it as a personalized mini-film.",
            "Fridge (advertisement): Fridge is a 2006 television and print advertising campaign launched by Diageo to promote canned Guinness-brand stout in the United Kingdom.",
            "Think Small: Think Small was one of the most famous ads in the advertising campaign for the Volkswagen Beetle, art directed by Helmut Krone.",
            "Grrr (advertisement): Grrr was a 2004 advertising campaign launched by Honda to promote its newly launched i-CTDi diesel engines in the United Kingdom.",
            "Four Million Smiles: Four Million Smiles was the theme of an advertising campaign in Singapore by the Singapore 2006 Organising Committee, sponsored by the government of Singapore.",
            "Diageo: Diageo plc ( or ) is a British multinational alcoholic beverages company, with its headquarters in London, England.",
            "The Trillion Dollar Campaign: The Trillion Dollar Campaign is an outdoor advertising campaign launched in 2009 to promote the newspaper \"The Zimbabwean\" in South Africa.",
            "Marcus Rivers: Marcus Rivers (portrayed by child-actor Bobb'e J. Thompson) is a fictional 12-year-old character that was used by Sony Computer Entertainment America as part of their \"Step Your Game Up\" advertising campaign for the PlayStation Portable and PSPgo consoles in North America, much like the PlayStation 3's \"It Only Does Everything\" advertising campaign commercials with Kevin Butler.",
            "Make a Smellmitment: Make a Smellmitment is an advertising campaign created by Wieden+Kennedy for Old Spice in the United States.",
        ],
        metadata={
            "id": "5a7773be5542997042120a51",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Fridge (advertisement)", 0], ["Diageo", 0]],
        },
    ),
    _RawExample(
        question="John Lineker, the mixed martial artist, is named after which English footballer and current sports broadcaster?",
        answer="Gary Winston Lineker",
        context=[
            "Mohammed &quot;The Hawk&quot; Shahid: Mohammed Shahid (born July 8, 1989) is an entrepreneur and a mixed martial artist from Bahrain.",
            "Weeshie Fogarty: Aloysius Fogarty (born March 1941) better known as Weeshie Fogarty, is an Irish retired Gaelic footballer, referee and current sports broadcaster.",
            "Akihiko Adachi: Akihiko Adachi (\u5b89\u9054 \u660e\u5f66 , Adachi Akihiko ) is a Japanese light heavyweight mixed martial artist.",
            "Colm O'Rourke: Colm O'Rourke (born 31 August 1957) is a retired Gaelic footballer and current sports broadcaster.",
            "Juan Pablo Sor\u00edn: Juan Pablo Sor\u00edn (born 5 May 1976) is an Argentine former footballer and current sports broadcaster, who played as a left back or left midfielder.",
            "John Lineker: John Lineker dos Santos de Paula (born January 1, 1990) known professionally as John Lineker, is a Brazilian mixed martial artist who currently competes in the Bantamweight division of the Ultimate Fighting Championship.",
            "M\u00edche\u00e1l \u00d3 Cr\u00f3in\u00edn: M\u00edche\u00e1l \u00d3 Cr\u00f3in\u00edn (born 1977) is an Irish retired Gaelic footballer and current sports broadcaster.",
            "Gary Lineker: Gary Winston Lineker, OBE ( ; born 30 November 1960) is an English retired footballer and current sports broadcaster.",
            "Pat Spillane: Patrick Gerard Spillane (born 1 December 1955), better known as Pat Spillane, is an Irish retired Gaelic footballer and current sports broadcaster.",
            "Pat Miletich: Patrick Jay \"Pat\" Miletich ( ; born March 9, 1966) is a retired American mixed martial artist and a current sports commentator.",
        ],
        metadata={
            "id": "5a8991ec5542993b751ca948",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["John Lineker", 0], ["John Lineker", 1], ["Gary Lineker", 0]],
        },
    ),
    _RawExample(
        question="What area is the species of Spruce which can be found in Rensselaer Plateau native to?",
        answer="eastern North America",
        context=[
            "Rensselaer Plateau: The Rensselaer Plateau is a small plateau located in the central portion of Rensselaer County, New York; it generally encompasses significant parts of the towns of Berlin, Stephentown, Sand Lake, Poestenkill, and Grafton, along with small sections of several other nearby towns.",
            "Big Bowman Pond: Big Bowman Pond is a small glacial lake in the Taborton section of the Town of Sand Lake, Rensselaer County, New York, United States.",
            "Round Pond (Berlin, New York): Round Pond is a small glacial lake in the Town of Berlin, Rensselaer County, New York, United States.",
            "Spring Lake (Berlin, New York): Spring Lake is a small glacial lake in the Town of Berlin, Rensselaer County, New York, United States.",
            "Spruce sawflies: Spruce sawflies are various sawfly species found in North America that attack spruce.",
            "Blue spruce: The blue spruce, green spruce, white spruce, Colorado spruce, or Colorado blue spruce, with the scientific name Picea pungens, is a species of spruce tree.",
            "Little Bowman Pond: Little Bowman Pond is a small glacial lake in the Taborton section of the Town of Sand Lake, Rensselaer County, New York, United States.",
            "Picea rubens: Picea rubens, commonly known as red spruce, is a species of spruce native to eastern North America, ranging from eastern Quebec to Nova Scotia, and from New England south in the Adirondack Mountains and Appalachians to western North Carolina.",
            "Picea glauca: Picea glauca, the white spruce, is a species of spruce native to the northern temperate and boreal forests in North America.",
            "Cinara pilicornis: Cinara pilicornis, the spruce shoot aphid or brown spruce shoot aphid, is an aphid species in the genus \"Cinara\" found on Norway spruce (\"Picea abies\") and Sitka spruce (\"Picea sitchensis\").",
        ],
        metadata={
            "id": "5a83143655429954d2e2ec0a",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Rensselaer Plateau", 3], ["Picea rubens", 0]],
        },
    ),
    _RawExample(
        question="Ammonium perchlorate was involved in the 1988 disaster that took place in what Nevada city?",
        answer="Henderson",
        context=[
            "Music In The Mountains: Since 1982, Music in the Mountains has been a summer classical music festival that takes place in Nevada County, California.",
            "Graniteville, California: Graniteville (previously: Eureka and unofficially Eureka South) is a small, unincorporated community and census-designated place (CDP) located in Nevada County, California, United States.",
            "Gary Goldschneider: Gary Goldschneider (born 22 May 1939) is a writer, pianist and composer.",
            "Texel Disaster: The Texel Disaster took place off the Dutch coast on the night of 31 August 1940 and involved the sinking of two Royal Navy destroyers, and damage to a third and a light cruiser.",
            "Camptonville, California: Camptonville (formerly, Comptonville and Gold Ridge) is a small town and census-designated place (CDP) located in northeastern Yuba County, California.",
            "Ammonium perchlorate: Ammonium perchlorate (\"AP\") is an inorganic compound with the formula NHClO.",
            "For-Site Foundation: The For-Site Foundation, established in 2003, is a non-profit organization dedicated to the creation of collaborative art about place.",
            "Alder Gulch: Alder Gulch (alternatively called Alder Creek) is a place in the Ruby River valley, in the U.S. state of Montana, where gold was discovered on May 26, 1863 by William Fairweather and a group of men including Barney Hughes, Thomas Cover, Henry Rodgers, Henry Edgar and Bill Sweeney who were returning to the gold fields of Grasshopper Creek, Bannack, Montana.",
            "PEPCON disaster: The PEPCON disaster was an industrial disaster that occurred in Henderson, Nevada on May 4, 1988 at the Pacific Engineering and Production Company of Nevada (PEPCON) plant.",
            "2009 Shelby 427: The 2009 Shelby 427 was the third race of the 2009 NASCAR Sprint Cup season.",
        ],
        metadata={
            "id": "5add58dd5542990dbb2f7e4a",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Ammonium perchlorate", 4], ["PEPCON disaster", 0]],
        },
    ),
    _RawExample(
        question="Who was born first Ursula Hegi or Beatrix Potter?",
        answer="Beatrix Potter",
        context=[
            "Hill Top, Cumbria: Hill Top is a 17th-century house in Near Sawrey near Hawkshead, in the English county of Cumbria.",
            "The Tales of Beatrix Potter: The Tales of Beatrix Potter (US title: \"Peter Rabbit and Tales of Beatrix Potter\") is a 1971 ballet film based on the children's stories of English author and illustrator Beatrix Potter.",
            "Stones from the River: Stones from the River is the 1994 novel by Ursula Hegi, and was chosen as an Eagles selection in February 1997.",
            "Beatrix Potter Gallery: The Beatrix Potter Gallery is a gallery run by the National Trust and situated in a 17th-century stone-built house in Hawkshead, Cumbria, England.",
            "Miss Potter: Miss Potter is a 2006 Anglo-American biographical fiction family drama film directed by Chris Noonan.",
            "The Tales of Beatrix Potter (ballet): The Tales of Beatrix Potter is a 1992 ballet adapted for stage by Anthony Dowell from a 1971 film that was choreographed by Frederick Ashton that in turn was based on the children's books by Beatrix Potter.",
            "The Fairy Caravan: The Fairy Caravan is a children's book written and illustrated by Beatrix Potter and first published in 1929 by Alexander McKay in Philadelphia, as Beatrix Potter did not wish to publish the book in UK, presuming it would not appeal to a British audience.",
            "Ursula Hegi: Ursula Hegi (born May 23, 1946) is a German-born American writer.",
            "The Tale of Two Bad Mice: The Tale of Two Bad Mice is a children's book written and illustrated by Beatrix Potter, and published by Frederick Warne & Co.",
            "Beatrix Potter: Helen Beatrix Potter (28 July 186622 December 1943) was an English writer, illustrator, natural scientist, and conservationist best known for her children's books featuring animals, such as those in \"The Tale of Peter Rabbit\".",
        ],
        metadata={
            "id": "5a72a2da5542991f9a20c548",
            "difficulty": "medium",
            "reasoning_type": "comparison",
            "supporting_facts": [["Ursula Hegi", 0], ["Beatrix Potter", 0]],
        },
    ),
    _RawExample(
        question="Which 2009 film features an actor that also started in Bad Samaritan?",
        answer="Cherrybomb",
        context=[
            "Bad Girl (Meisa Kuroki song): \"Bad Girl\" is a song by Japanese recording artist Meisa Kuroki from her debut extended play (EP), \"Hellcat\".",
            "Bad Samaritan (film): Bad Samaritan is an upcoming American crime thriller film directed by Dean Devlin and written by Brandon Boyce.",
            "Kawan Prather: Kawan \"KP\" Prather is an American record executive, songwriter, record producer and member of first generation Dungeon Family group P.A..",
            "Harry Stubbs (actor): Harry Oakes Stubbs (December 7, 1874 \u2013 May 9, 1950) was an English-born American character actor, who appeared both on Broadway and in films.",
            "Robert Sheehan: Robert Michael Sheehan (Irish: \"Roibe\u00e1rd M\u00edche\u00e1l \u00d3 Siodhach\u00e1in\" ; born 7 January 1988) is an Irish actor.",
            "Kwan Hoi-san: Herman Kwan Hoi-San () (October 23, 1925 in Guangzhou, Guangdong \u2014 September 11, 2006) was a Hong Kong actor.",
            "Star Trek Into Darkness: Star Trek Into Darkness is a 2013 American science fiction action film directed by J. J. Abrams and written by Roberto Orci, Alex Kurtzman, and Damon Lindelof.",
            "Charles Wadsworth: Charles Wadsworth is a classical pianist and musical promoter from Newnan, Georgia.",
            "Metropolitan Fiber Systems: Metropolitan Fiber Systems Inc, later known as MFS Communications Company, was a last mile provider of business grade telecommunication products such as long distance, and Internet access through its own fiber rings in major central business districts throughout North America and Europe.",
            "BumRush: BumRush is a 2011 Canadian film directed by Michel Jett\u00e9.",
        ],
        metadata={
            "id": "5ae0f4fe55429945ae959491",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Bad Samaritan (film)", 1], ["Robert Sheehan", 1]],
        },
    ),
    _RawExample(
        question="Name the German state whose capital is known for a bombing raid during the second world war where 3,900 tons of high-explosive bombs and incendiary devices were dropped with the help of Sir Arthur Harris Air Officer Commanding-in-Chief?",
        answer="Saxony",
        context=[
            "Bombing of Dresden in World War II: The bombing of Dresden was a British/American aerial bombing attack on the city of Dresden, the capital of the German state of Saxony, that took place during the Second World War in the European Theatre.",
            "Thomas Warne-Browne: Air Marshal Sir Thomas Arthur Warne-Browne, (21 July 1898 \u2013 13 October 1962) was a senior Royal Air Force officer who served as Air Officer Commanding-in-Chief Maintenance Command from 1949 to 1952.",
            "Arthur Coningham (RAF officer): Air Marshal Sir Arthur \"Mary\" Coningham, {'1': \", '2': \", '3': \", '4': \"} (19 January 1895 \u2013 presumably 30 January 1948) was a senior officer in the Royal Air Force.",
            "Leonard Slatter: Air Marshal Sir Leonard Horatio Slatter, (8 December 1894 \u2013 14 April 1961) was a naval aviator during the First World War and a senior Royal Air Force commander during the Second World War.",
            "Sir Arthur Harris, 1st Baronet: Marshal of the Royal Air Force Sir Arthur Travers Harris, 1st Baronet, {'1': \", '2': \", '3': \", '4': \"} (13 April 1892 \u2013 5 April 1984), commonly known as \"Bomber\" Harris by the press and often within the RAF as \"Butcher\" Harris, was Air Officer Commanding-in-Chief (AOC-in-C) RAF Bomber Command during the height of the Anglo-American strategic bombing campaign against Nazi Germany in the Second World War.",
            "Bombing of Hanover in World War II: The Bombing of Hannover was a series of eighty-eight air raids by RAF Bomber Command and the United States Army Air Forces on the German city of Hannover during World War II.",
            "Lawrence Pattinson: Air Marshal Sir Lawrence Arthur Pattinson, (8 October 1890 \u2013 28 March 1955) was a Royal Air Force officer who became Air Officer Commanding-in-Chief of Flying Training Command from 1940 to 1941.",
            "Bombing of Gorky in World War II: The bombing of Gorky by the Luftwaffe continued from 1941 to 1943 in the Eastern Front theatre of World War II.",
            "Philip Babington: Air Marshal Sir Philip Babington, (25 February 1894 \u2013 25 February 1965) was a Royal Air Force officer who served as Air Officer Commanding-in-Chief of Flying Training Command from 1942 to 1945 during the Second World War.",
            "Denis Barnett: Air Chief Marshal Sir Denis Hensley Fulton Barnett, {'1': \", '2': \", '3': \", '4': \"} (11 February 1906 \u2013 31 December 1992) was a squadron commander and senior officer in the Royal Air Force during the Second World War.",
        ],
        metadata={
            "id": "5ae7161a5542995703ce8bcc",
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["Sir Arthur Harris, 1st Baronet", 0], ["Sir Arthur Harris, 1st Baronet", 3], ["Bombing of Dresden in World War II", 0], ["Bombing of Dresden in World War II", 1]],
        },
    ),
    _RawExample(
        question="What is the oldest age, in years, that Selim Zilkha's retailer caters to for children? ",
        answer="8",
        context=[
            "Selim Zilkha: Selim Zilkha (born 1927) is an Iraqi-born British entrepreneur, who founded Mothercare, one of the UK's largest retail chains.",
            "Ieper Group: The Ieper Group (Dutch: \"Ieper Groep\" ; French: \"Groupe d'Ypres\" ) is a group of rock strata in the subsurface of northwest Belgium.",
            "Tournaisian: The Tournaisian is in the ICS geologic timescale the lowest stage or oldest age of the Mississippian, the oldest subsystem of the Carboniferous.",
            "Mothercare: Mothercare plc () is a British retailer which specialises in products for expectant mothers and in general merchandise for children up to 8 years old.",
            "Ukrainian Youth Football League: The Ukrainian National Youth Competition (Ukrainian: \"\u0414\u0438\u0442\u044f\u0447\u043e-\u044e\u043d\u0430\u0446\u044c\u043a\u0430 \u0444\u0443\u0442\u0431\u043e\u043b\u044c\u043d\u0430 \u043b\u0456\u0433\u0430 \u0423\u043a\u0440\u0430\u0457\u043d\u0438\", (\u0414\u042e\u0424\u041b\u0423) ) is split into four (4) age groups which in turn has the Supreme and the First Leagues.",
            "Selim Aga: Selim Aga (born around 1826 in Taqali area of Sudan, died December 1875 in Liberia), a native of Sudan who was abducted by slave traders when he was eight years of age, was brought to Scotland in 1836, and raised and educated as a free man.",
            "Hesham Selim: Hesham Selim (Arabic: \u0647\u0634\u0627\u0645 \u0633\u0644\u064a\u0645\u200e \u200e ) is an Egyptian actor and the son of Saleh Selim.",
            "Aquitanian (stage): The Aquitanian is, in the ICS' geologic timescale, the oldest age or lowest stage in the Miocene.",
            "Ypresian: In the geologic timescale the Ypresian ( ) is the oldest age or lowest stratigraphic stage of the Eocene.",
            "Danian: The Danian is the oldest age or lowermost stage of the Paleocene epoch or series, the Paleogene period or system and the Cenozoic era or erathem.",
        ],
        metadata={
            "id": "5adc4f7855429947ff173922",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Selim Zilkha", 0], ["Mothercare", 0]],
        },
    ),
    _RawExample(
        question="Which of these programs has already launched, Astron or James Webb Space Telescope?",
        answer="Astron",
        context=[
            "Astron (spacecraft): Astron was a Soviet spacecraft launched on 23 March 1983 at 12:45:06 UTC, using Proton launcher, which was designed to fulfill an astrophysics mission.",
            "X-ray Astronomy Recovery Mission: The X-ray Astronomy Recovery Mission (XARM, pronounced \"charm\") is an X-ray astronomy satellite project of the Japan Aerospace Exploration Agency (JAXA) to provide breakthroughs in the study of structure formation of the universe, outflows from galaxy nuclei, and dark matter.",
            "Integrated Science Instrument Module: Integrated Science Instrument Module (ISIM) is a component of the James Webb Space Telescope, a large international infrared space telescope planned for launch in 2018.",
            "New Worlds Mission: The New Worlds Mission is a proposed project comprising a large occulter in space designed to block the light of nearby stars in order to observe their orbiting exoplanets.",
            "Spacecraft Bus (JWST): Spacecraft Bus is the primary support component of the James Webb Space Telescope, that hosts a multitude of computing, communication, propulsion, and structural parts, bringing the different parts of the telescope together.",
            "Sunshield (JWST): Sunshield is a component of the James Webb Space Telescope, designed to shield the main optics from the Sun's heat and light.",
            "Space Telescope Science Institute: The Space Telescope Science Institute (STScI) is the science operations center for the Hubble Space Telescope (HST; in orbit since 1990) and for the James Webb Space Telescope (JWST; scheduled to be launched in 2018).",
            "Fine guidance sensor: A fine guidance sensor (FGS) is an instrument on board a space telescope that provides high-precision pointing information as input to the observatory's attitude control systems.",
            "Optical Telescope Element: Optical Telescope Element (OTE) is a sub-section of the James Webb Space Telescope, a large infrared space telescope scheduled to be launched in October 2018.",
            "James Webb Space Telescope: The James Webb Space Telescope (JWST) is a space telescope that is part of NASA's Next Generation Space Telescope program, developed in coordination between NASA, the European Space Agency, and the Canadian Space Agency.",
        ],
        metadata={
            "id": "5ab554fd554299488d4d9942",
            "difficulty": "hard",
            "reasoning_type": "comparison",
            "supporting_facts": [["Astron (spacecraft)", 0], ["James Webb Space Telescope", 1]],
        },
    ),
    _RawExample(
        question="Which US Supreme Court Case occurred first, Gravel v. United States or Schenck v. United States?",
        answer="Schenck v. United States",
        context=[
            "Schenck v. United States: Schenck v. United States, 249 U.S. 47 (1919) , is a United States Supreme Court case concerning enforcement of the Espionage Act of 1917 during World War I.",
            "Stephen Halbrook: Stephen P. Halbrook is a Senior Fellow at the Independent Institute and an author and lawyer known for his litigation on behalf of the National Rifle Association.",
            "Oliver Wendell Holmes Jr.: Oliver Wendell Holmes Jr. ( ; March 8, 1841 \u2013 March 6, 1935) was an American jurist who served as an Associate Justice of the Supreme Court of the United States from 1902 to 1932, and as Acting Chief Justice of the United States from January\u2013February 1930.",
            "Imminent lawless action: \"Imminent lawless action\" is a standard currently used that was established by the United States Supreme Court in \"Brandenburg v. Ohio\" (1969), for defining the limits of freedom of speech.",
            "Freedom for the Thought That We Hate: Freedom for the Thought That We Hate: A Biography of the First Amendment is a 2007 non-fiction book by journalist Anthony Lewis about freedom of speech, freedom of the press, freedom of thought, and the First Amendment to the United States Constitution.",
            "Brandenburg v. Ohio: Brandenburg v. Ohio, 395 U.S. 444 (1969) , was a landmark United States Supreme Court case based on the First Amendment to the U.S. Constitution.",
            "Gravel v. United States: Gravel v. United States, 408 U.S. 606 (1972), was a case regarding the protections offered by the Speech or Debate Clause of the United States Constitution.",
            "Charles Schenck: Charles T. Schenck was the secretary of the Socialist Party of America in Philadelphia during the First World War and involved in the 1919 Supreme Court case \"Schenck v. United States\".",
            "Rodr\u00edguez v. Popular Democratic Party: Rodr\u00edguez v. Popular Democratic Party, 457 U.S. 1 (1982) , was a case in which the Supreme Court of the United States heard on appeal from the Supreme Court of Puerto Rico whether Puerto Rico may by statute vest in a political party the power to fill an interim vacancy in the Puerto Rico Legislature.",
            "Shouting fire in a crowded theater: \"Shouting \"fire\" in a crowded theater\" is a popular metaphor for speech or actions made for the principal purpose of creating unnecessary panic.",
        ],
        metadata={
            "id": "5ab5994e554299488d4d99f3",
            "difficulty": "medium",
            "reasoning_type": "comparison",
            "supporting_facts": [["Gravel v. United States", 0], ["Schenck v. United States", 0]],
        },
    ),
    _RawExample(
        question="Which director directed more films, Sydney Pollack or Millard Webb?",
        answer="Sydney Irwin Pollack",
        context=[
            "William Steinkamp: William Steinkamp (born June 9, 1953) is an American film editor with more than 20 film credits.",
            "Sketches of Frank Gehry: Sketches of Frank Gehry is a 2006 American documentary film directed by Sydney Pollack and produced by Ultan Guilfoyle, about the life and work of the Canadian-American architect Frank Gehry.",
            "The Drop Kick: The Drop Kick (also known as \"Glitter\" in the UK) is a 1927 silent film directed by Millard Webb written by Katherine Brush about a college football player (Richard Barthelmess) who finds his reputation on the line when he pays an innocent visit to a woman whose husband kills himself.",
            "The Sea Beast: The Sea Beast is a 1926 American silent drama film directed by Millard Webb, starring John Barrymore and Dolores Costello.",
            "Reaching for the Moon (1917 film): Reaching for the Moon is a 1917 American silent adventure film directed by John Emerson and written by John Emerson, Joseph Henabery, and Anita Loos.",
            "The Happy Ending (1931 film): The Happy Ending is a 1931 British drama film directed by Millard Webb and starring George Barraud, Daphne Courtney and Alfred Drayton.",
            "Sydney Pollack: Sydney Irwin Pollack (July 1, 1934 \u2013 May 26, 2008) was an American film director, producer and actor.",
            "Millard Webb: Millard Webb (6 December 1893 \u2013 21 April 1935), was an American screenwriter and director.",
            "Hearts of Youth: Hearts of Youth is a 1921 American silent film based on the novel \"Ishmael\" by E. D. E. N. Southworth.",
            "The Love Thrill: The Love Thrill is a lost 1927 silent film comedy directed by Millard Webb and starring Laura La Plante and Tom Moore.",
        ],
        metadata={
            "id": "5a7d7e8a5542995f4f402283",
            "difficulty": "medium",
            "reasoning_type": "comparison",
            "supporting_facts": [["Sydney Pollack", 0], ["Sydney Pollack", 1], ["Millard Webb", 1]],
        },
    ),
    _RawExample(
        question="The album titled Omo Baba Olowo by Davido was produced by who?",
        answer="Maleek Berry",
        context=[
            "GospelOnDeBeatz: Gospelondebeatz also known as Gospel Chinemeremu Obi (born; January 14, 1987) is a Nigerian born Music Producer and songwriter.",
            "Runtown discography: Nigerian recording artist Runtown has released one studio album, seventeen singles and ten music videos. His debut single was released in 2007 as an upcoming artist.",
            "The Headies 2012: The seventh edition of The Headies (formerly called The Hip Hop World Awards) was hosted by M.I and Omawumi.",
            "Omo Baba Olowo: Omo Baba Olowo (Yoruba: \"Son of a Rich Man\"), stylized as Omo Baba Olowo: The Genesis or simply O.B.O, is the debut studio album by Nigerian recording artist and record producer Davido.",
            "Live: Sadler's Wells: Pete Townshend Live: Sadler's Wells 2000 is a live album released by Pete Townshend in 2000.",
            "Maleek Berry: Maleek Shoyebi, popularly known as Maleek Berry, is a British-born Nigerian record producer and recording artist.",
            "Gallardo (Runtown song): \"Gallardo\" (pronounced ( ; ] ) is a song by Nigerian Afropop recording artist Runtown, released from his upcoming debut studio album titled \"Ghetto University\".",
            "Dami Duro: \"Dami Duro\" is a song by Nigerian recording artist Davido.",
            "Lagbaja: According to L\u00e1gb\u00e1j\u00e1 (pronounced la gba jah), his mask is used as an icon of man's facelessness.",
            "In Death Reborn: On February 11, 2014, it was confirmed that the album's production team consisted of producers include Stu Bangas, C-Lance, Leaf Dog, Panik and including Army of the Pharaohs' own Apathy amongst others, including new faces that hadn't been producing for the group beforehand.",
        ],
        metadata={
            "id": "5add8e425542997545bbbd7b",
            "difficulty": "hard",
            "reasoning_type": "bridge",
            "supporting_facts": [["Omo Baba Olowo", 2], ["Maleek Berry", 2]],
        },
    ),
    _RawExample(
        question="What county in New Hampshire did Alex Preston, the third place contestant on the thirteeth season of \"American Idol\", come from?",
        answer="Hillsborough County",
        context=[
            "DialIdol: DialIdol is both the name of a computer program for Microsoft Windows and its associated website that tracks voting trends for \"American Idol\" contestants.",
            "Mayr\u00e9 Mart\u00ednez: Mayr\u00e9 Andrea de los \u00c1ngeles Mart\u00ednez Blanco (Born November 28 in Caracas, Venezuela), is a Latin pop singer, songwriter.",
            "Melinda Doolittle: Melinda Marie Doolittle (born October 6, 1977) is an American singer who finished as the third place finalist on the sixth season of \"American Idol\".",
            "Anoop Desai: Anoop Manoj Desai (born December 20, 1986) is an American singer-songwriter best known for his time as a contestant on the eighth season of \"American Idol\".",
            "Jordin Sparks: Jordin Brianna Sparks (born December 22, 1989) is an American singer, songwriter, and actress.",
            "Vote for the Worst: VoteForTheWorst.com (VFTW) was a website devoted to voting for the worst, most entertaining, most hated or quirkiest contestants on the Fox Network television series \"American Idol\" as well as the NBC Network television series \"The Voice\".",
            "Alex Preston (singer): Alex Preston Philbrick (born May 6, 1993), better known as Alex Preston, is an American singer from Mont Vernon, New Hampshire, who was a finalist on the thirteenth season of \"American Idol\", coming in third place.",
            "Mandisa: Mandisa Lynn Hundley (born October 2, 1976), known professionally as Mandisa, is an American gospel and contemporary Christian recording artist.",
            "New Zealand Idol: NZ Idol, more commonly known as New Zealand Idol, was the New Zealand version of the Idol series originated as the hit British TV series \"Pop Idol\".",
            "Mont Vernon, New Hampshire: Mont Vernon is a town in Hillsborough County, New Hampshire, United States.",
        ],
        metadata={
            "id": "5a8c480f5542995e66a47593",
            "difficulty": "easy",
            "reasoning_type": "bridge",
            "supporting_facts": [["Alex Preston (singer)", 0], ["Mont Vernon, New Hampshire", 0]],
        },
    ),
    _RawExample(
        question="Who directed the upcoming movie that Robinne Lee appears in?",
        answer="James Foley",
        context=[
            "The Theatre Army Productions: The Theatre Army Productions is a North (Punjab/Chandigarh) based production house which was founded by Gaurav Sharma who is usually known as Gabbar.",
            "Shamata Anchan: Shamata Anchan is a model and Indian television actress.",
            "Sabesh-Murali: Sabesh-Murali is an Indian musical duo, consisting of two Tamil music directors and playback singers who have jointly composed for many Tamil films in Chennai, India.",
            "Too Hot to Handle (1960 film): Too Hot to Handle (released in the United States as Playgirl After Dark) is a 1960 British neo-noir gangster thriller film, starring Jayne Mansfield and Leo Genn.",
            "Kentucky Jones: Kentucky Jones is a half-hour comedy/drama starring Dennis Weaver as Kenneth Yarborough \"K.Y. or Kentucky\" Jones, D.V.M., a recently widowed former horse trainer and active horse farm owner, who becomes the guardian of Dwight Eisenhower \"Ike\" Wong, a 10-year-old Chinese orphan, played by Ricky Der.",
            "Peter Lee (cricketer): Peter Granville Lee, affectionately known as \"Leapy\", born at Arthingworth, Northamptonshire, on 27 August 1945, is a former cricketer who played for Northamptonshire and Lancashire.",
            "Hercules in the Haunted World: Hercules in the Haunted World (Italian: Ercole al centro della terra) is a 1961 Italian sword-and-sandal film directed by Mario Bava.",
            "Fifty Shades Freed (film): Fifty Shades Freed is an upcoming American erotic romantic drama film directed by James Foley and written by Niall Leonard, based on the novel of same name by E. L. James.",
            "Main Solah Baras Ki: Main Solah Baras Ki is a Bollywood film directed by Dev Anand and was released in 1998.",
            "Robinne Lee: Robinne Lee (born July 16, 1974) is an American actress and author ab.",
        ],
        metadata={
            "id": "5ab911f55542991b5579f0dc",
            "difficulty": "medium",
            "reasoning_type": "bridge",
            "supporting_facts": [["Robinne Lee", 1], ["Fifty Shades Freed (film)", 0]],
        },
    ),

]

_DATASET: Dataset | None = None
_DATASETS_DIR = Path(__file__).parent / "datasets"


def dataset_path() -> Path:
    """Return the path to the bundled HotpotQA dataset file."""
    return _DATASETS_DIR / "hotpotqa_dev_subset.jsonl"


def _build_examples(raw_examples: Iterable[_RawExample]) -> list[EvaluationExample]:
    """Convert raw examples to EvaluationExample format."""
    examples: list[EvaluationExample] = []
    for raw in raw_examples:
        metadata = dict(raw.metadata)
        metadata.setdefault("case_study", "hotpotqa")
        metadata.setdefault("fallback_answer", raw.answer)
        examples.append(
            EvaluationExample(
                input_data={"question": raw.question, "context": raw.context},
                expected_output=raw.answer,
                metadata=metadata,
            )
        )
    return examples


def _ensure_dataset_file() -> None:
    """Generate the JSONL dataset file if it doesn't exist."""
    path = dataset_path()
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for raw in _RAW_EXAMPLES:
            record = {
                "input": {"question": raw.question, "context": raw.context},
                "output": raw.answer,
                "metadata": raw.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_official_hotpotqa(file_path: Path, max_examples: int = 100) -> list[_RawExample]:
    """Load examples from the official HotpotQA JSON file.

    Args:
        file_path: Path to hotpot_dev_distractor_v1.json
        max_examples: Maximum number of examples to load (default 100 for demo)

    Returns:
        List of _RawExample objects
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    examples: list[_RawExample] = []
    for item in data[:max_examples]:
        # HotpotQA format: each item has 'context' as list of [title, sentences]
        context_passages: list[str] = []
        for title, sentences in item.get("context", []):
            # Combine title with first sentence for context
            if sentences:
                context_passages.append(f"{title}: {sentences[0]}")

        examples.append(
            _RawExample(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                context=context_passages,
                metadata={
                    "id": item.get("_id", ""),
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown"),
                    "supporting_facts": item.get("supporting_facts", []),
                },
            )
        )

    return examples


def load_case_study_dataset(
    *, force_refresh: bool = False, max_examples: int | None = None
) -> Dataset:
    """Return a cached dataset containing HotpotQA multi-hop questions.

    If HOTPOTQA_DATASET_PATH environment variable is set, loads from that file.
    Otherwise uses the bundled 50-example dataset.

    Args:
        force_refresh: Force reload of the dataset
        max_examples: Maximum examples to load from full benchmark (default: 100)

    Returns:
        Dataset object with HotpotQA examples
    """
    global _DATASET

    external_path = os.environ.get(HOTPOTQA_DATASET_PATH_ENV)

    if _DATASET is None or force_refresh:
        if external_path:
            # Load from official HotpotQA benchmark
            path = Path(external_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"HotpotQA dataset not found at {path}. "
                    f"Download from: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
                )
            raw_examples = _load_official_hotpotqa(path, max_examples or 100)
            description = f"Official HotpotQA benchmark ({len(raw_examples)} examples)"
        else:
            # Use bundled examples
            _ensure_dataset_file()
            raw_examples = list(_RAW_EXAMPLES)
            description = "Bundled HotpotQA examples (50 questions, train split)"

        _DATASET = Dataset(
            examples=_build_examples(raw_examples),
            name="hotpotqa_case_study",
            description=description,
            metadata={"task": "multi_hop_qa", "source": "official" if external_path else "bundled"},
        )

    return _DATASET
