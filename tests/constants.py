# tests/constants.py

# constant variables
NINF = float("-inf")
MAX_NEW_TOKEN_NUM = 40
MAX_MODEL_LEN = 512

# benchmark
SHARE_GPT_FILE_NAME = "../models/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json"
SHARE_GPT_REQ_NUM = 128
SHARE_GPT_MAX_SEQS = 8
MAX_NUM_BLOCKS = 2048

DATA_DIR = "./data"
LOG_DIR = "./log"

# sampler
REP_PEN_OFF = 1.
TEMP_GREEDY = 0.

TOK = 100
TOK_EOS = 151643

# prompt
PROMPT_CLASSICAL = "The capital of France is"
PROMPT_BATCH_1 = [PROMPT_CLASSICAL]
PROMPT_BATCH_2 = [PROMPT_CLASSICAL, "Hi"]

LONG_PROMPT_1 = ("The capital of France is Paris. It's the largest city in Europe, and it's also one of the most important cities in the world."
"Paris has a very long history dating back to ancient times. The first inhabitants were the Gauls who lived here for many years until the Romans arrived around 450 BC. The Romans built an impressive amphitheatre called Colosseum which was used as a place where gladiators fought each other or animals were sacrificed."
"After the fall of Rome in the 7th century AD, Paris became a major center for trade with Asia Minor (modern-day Turkey) and the Mediterranean Sea. In 1358, King Philip II of France conquered the city and made it his own. The French king, Charles V, later moved the royal court to Paris."
)

LONG_PROMPT_2 = ("The capital of France is Paris. It is the largest city in Europe and the third largest city in the world. "
    "It is located in the south of France, on the banks of the Seine River. It is situated on the Île de la Cité, which is a small island in the center of the city. "
    "The city is surrounded by the Seine River and the Mediterranean Sea. It is also surrounded by the Pyrenees mountains. "
    "The city is known for its beautiful architecture, its rich history, and its beautiful parks and gardens. "
)

LONG_BATCH = [LONG_PROMPT_1,]
LONG_BATCH_2 = [LONG_PROMPT_1, LONG_PROMPT_2]


BATCH_FOR_BENCHMARKING = [
    PROMPT_CLASSICAL,
    "Hi",
    "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...",
    "How are you?",
    "How to tell if a customer segment is well segmented? In 3 bullet points.",
    "Where are you from?",
    "Do you know the book Traction by Gino Wickman",
    "create new version. we will call it: \"second draft\". You need to reformat Filters part to be more ease to read",
    "test: [noun] a means of testing: such as. something (such as a series of questions or exercises) for measuring the skill, knowledge, intelligence, capacities, or aptitudes of an individual or group. a procedure, reaction, or reagent used to identify or characterize a substance or constituent. a positive result in such a test.",
    "what is a good maven pom.xml template for compiling a java project?",
    "What is the most quoted biblical verse?",
    "how to use case-sensative sorting in js?",
    "explain the process of a product designer",
    "What is the nature of reality?",
    "generate apache airflow configuration code for supporting azure active directory sso tokens",
    "What is the nature of the universe?",
    "i need a copy paste tool for zellij. its relaly hard to grab scrollback output cleanly, idk how to output it easily for use outside the terminal. can you help? either a tool or just commands that allow me to extract that data cleanly wtihout jumping through hoops",
    "What is the nature of space?",
    '''Need you act as a senior developer. 
we use AWS, angular with EC2. We have a coaching tool that places images on screen of particpants and uses webRTC for communications, we will aso be adding lines from twilio for VOIP.

we currently have API for google and have trasncription working. and use GPT to analyze call. 

I have a few questions''',
    "What is the nature of matter?",
    "I have a food delivery business, I want ideas for menu items, like snacks, for 2 or more poeple to eat together, one I have in mind are mexican nachos",
    "What is the nature of light?",
    "What is the nature of gravity?",
    "1+1=?",
    "123*456=?",
]

