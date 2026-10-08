from graphviz import Digraph


# Load Brown Corpus Nouns

def load_nouns(filename):

    with open(filename, "r", encoding="utf-8") as f:

        return set(
            line.strip().lower()
            for line in f
            if line.strip()
        )


# Morphological FST

class MorphologicalFST:

    def __init__(self, nouns):

        self.nouns = nouns


        # States

        self.states = {

            "q0": "Start",

            "q_root": "Root Lookup",

            "q_s": "S Addition",

            "q_es": "E Insertion",

            "q_ies": "Y Replacement",

            "q_sg": "Singular Accept",

            "q_pl": "Plural Accept",

            "q_invalid": "Reject"

        }


        # Transition Table

        self.transitions = [

            ("q0",
             "noun root",
             "q_root",
             "ε"),


            # Singular

            ("q_root",
             "END",
             "q_sg",
             "+N+SG"),



            # S addition

            ("q_root",
             "+s",
             "q_s",
             "ε"),


            ("q_s",
             "valid root",
             "q_pl",
             "+N+PL"),


            ("q_s",
             "invalid root",
             "q_invalid",
             "Invalid Word"),



            # E insertion

            ("q_root",
             "+es",
             "q_es",
             "ε"),


            ("q_es",
             "root ends s,z,x,ch,sh",
             "q_pl",
             "+N+PL"),


            ("q_es",
             "invalid root",
             "q_invalid",
             "Invalid Word"),



            # Y replacement

            ("q_root",
             "+ies",
             "q_ies",
             "ε"),


            ("q_ies",
             "consonant+y root",
             "q_pl",
             "+N+PL"),


            ("q_ies",
             "invalid root",
             "q_invalid",
             "Invalid Word")

        ]



    # FST Parser

    def parse(self, word):

        word = word.lower()


        # 1. Y replacement
        # tries->try
        if word.endswith("ies"):
            root = word[:-3] + "y"

            if root in self.nouns:
                return root + "+N+PL"



        # 2. E insertion
        # foxes->fox
        if word.endswith("es"):
            root = word[:-2]

            if root in self.nouns and root.endswith(
                ("s","z","x","ch","sh")
            ):
                return root + "+N+PL"



        # 3. S addition
        # bags->bag
        if word.endswith("s"):

            root = word[:-1]

            if root in self.nouns:
                return root + "+N+PL"



        # 4. Singular
        if word in self.nouns:
            return word + "+N+SG"


        return "Invalid Word"


    # Print Transition Table

    def print_transition_table(self):

        print("\n")
        print("="*95)
        print("              MORPHOLOGICAL FST TRANSITION TABLE")
        print("="*95)


        print(
            f"{'Current State':20}"
            f"{'Input Symbol':30}"
            f"{'Next State':20}"
            f"{'Output'}"
        )


        print("-"*95)


        for state, inp, nxt, out in self.transitions:


            print(
                f"{state:20}"
                f"{inp:30}"
                f"{nxt:20}"
                f"{out}"
            )


        print("="*95)



    # Graphviz Diagram

    def generate_graph(self):

        dot = Digraph(
            "Morphological_FST",
            format="png"
        )


        dot.attr(rankdir="LR")


        # Start arrow

        dot.node(
            "start",
            "",
            shape="point"
        )


        # States

        for state,name in self.states.items():


            if state in ["q_sg","q_pl"]:

                dot.node(
                    state,
                    name,
                    shape="doublecircle"
                )

            else:

                dot.node(
                    state,
                    name,
                    shape="circle"
                )


        dot.edge(
            "start",
            "q0"
        )


        # Transitions


        dot.edge(
            "q0",
            "q_root",
            label="root"
        )


        dot.edge(
            "q_root",
            "q_sg",
            label="END / +N+SG"
        )


        dot.edge(
            "q_root",
            "q_s",
            label="+s"
        )


        dot.edge(
            "q_s",
            "q_pl",
            label="valid / +N+PL"
        )


        dot.edge(
            "q_root",
            "q_es",
            label="+es"
        )


        dot.edge(
            "q_es",
            "q_pl",
            label="valid / +N+PL"
        )


        dot.edge(
            "q_root",
            "q_ies",
            label="+ies"
        )


        dot.edge(
            "q_ies",
            "q_pl",
            label="valid / +N+PL"
        )


        dot.edge(
            "q_s",
            "q_invalid",
            label="invalid"
        )


        dot.edge(
            "q_es",
            "q_invalid",
            label="invalid"
        )


        dot.edge(
            "q_ies",
            "q_invalid",
            label="invalid"
        )


        dot.render(
            "brown_morphological_fst",
            view=True
        )


        print(
            "[+] Graph saved as brown_morphological_fst.png"
        )



# ==========================================================
# Main Program
# ==========================================================


nouns = load_nouns(
    "brown_nouns.txt"
)


fst = MorphologicalFST(nouns)



# Test cases

words = [

    "fox",
    "foxes",
    "foxs",

    "bag",
    "bags",

    "watch",
    "watches",

    "try",
    "tries"

]



print("\nFST OUTPUT")
print("-"*40)


for word in words:

    print(
        f"{word:10} = {fst.parse(word)}"
    )



# Print transition table

fst.print_transition_table()



# Generate Graphviz

fst.generate_graph()