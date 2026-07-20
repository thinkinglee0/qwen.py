# tests/constants.py

# constant variables
MAX_NEW_TOKEN_NUM = 40
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


_EPS = 1e-5
NINF = float("-inf")
