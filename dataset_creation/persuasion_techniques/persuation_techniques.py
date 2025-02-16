from dataclasses import dataclass
import random
from typing import List


@dataclass
class PersuasionTechnique:
    """
    Data class representing a persuasion technique with:
      - An ID (1–40)
      - A descriptive name
      - The broader strategy category
      - A flag indicating 'Ethical' vs. 'Unethical'
    """

    technique_id: int
    name: str
    strategy: str
    category: str  # e.g. "Ethical" or "Unethical"
    definition: str
    message_example: str


# Define all 40 persuasion techniques in a list
ALL_TECHNIQUES: List[PersuasionTechnique] = [
    # 1. Evidence-based Persuasion
    PersuasionTechnique(
        technique_id=1,
        name="Evidence-based Persuasion",
        strategy="Information-based",
        category="Ethical",
        definition='This technique is defined as "Supporting a claim with credible data, research, or facts."',
        message_example=(
            "Studies from the American Heart Association show that quitting smoking can reduce "
            "your risk of heart disease by half—would you consider stopping now for your health?"
        ),
    ),
    # 2. Logical Appeal
    PersuasionTechnique(
        technique_id=2,
        name="Logical Appeal",
        strategy="Information-based",
        category="Ethical",
        definition='This technique is defined as "Using reason and structured arguments to influence decisions."',
        message_example=(
            "If smoking damages your lungs and you need healthy lungs to live a longer life, "
            "it follows that quitting smoking will help you live longer."
        ),
    ),
    # 3. Expert Endorsement
    PersuasionTechnique(
        technique_id=3,
        name="Expert Endorsement",
        strategy="Credibility-based",
        category="Ethical",
        definition='This technique is defined as "Citing qualified specialists or professionals to validate a point."',
        message_example=(
            "Doctors at the Mayo Clinic emphasize that smoking cessation is one of the most "
            "effective steps to improve your overall health—would you consider quitting?"
        ),
    ),
    # 4. Non-expert Testimonial
    PersuasionTechnique(
        technique_id=4,
        name="Non-expert Testimonial",
        strategy="Credibility-based",
        category="Ethical",
        definition='This technique is defined as "Leveraging personal anecdotes or experiences from non-specialists."',
        message_example=(
            "My friend quit smoking last year, and she says her energy levels have skyrocketed—"
            "you could feel the same benefits if you stop too."
        ),
    ),
    # 5. Authority Endorsement
    PersuasionTechnique(
        technique_id=5,
        name="Authority Endorsement",
        strategy="Credibility-based",
        category="Ethical",
        definition='This technique is defined as "Invoking the status of an authoritative figure or institution to add credibility."',
        message_example=(
            "The Surgeon General repeatedly advises smokers to quit in order to prevent severe "
            "health complications—following this guidance could save your life."
        ),
    ),
    # 6. Social Proof
    PersuasionTechnique(
        technique_id=6,
        name="Social Proof",
        strategy="Norm-based",
        category="Ethical",
        definition='This technique is defined as "Demonstrating that many others already believe, choose, or behave in a certain way."',
        message_example=(
            "Most people in this neighborhood have already quit smoking—you can join them and be "
            "smoke-free too."
        ),
    ),
    # 7. Injunctive Norm
    PersuasionTechnique(
        technique_id=7,
        name="Injunctive Norm",
        strategy="Norm-based",
        category="Ethical",
        definition='This technique is defined as "Emphasizing what people are expected or encouraged to do by social standards."',
        message_example=(
            "Our community strongly discourages smoking for public health reasons—stopping would "
            "align with these shared expectations."
        ),
    ),
    # 8. Foot-in-the-door
    PersuasionTechnique(
        technique_id=8,
        name="Foot-in-the-door",
        strategy="Commitment-based",
        category="Ethical",
        definition='This technique is defined as "Starting with a small request to increase acceptance of a larger one later."',
        message_example=(
            "Could you commit to smoking one less cigarette a day? If that works, maybe you'll be "
            "ready to quit completely soon."
        ),
    ),
    # 9. Door-in-the-face
    PersuasionTechnique(
        technique_id=9,
        name="Door-in-the-face",
        strategy="Commitment-based",
        category="Ethical",
        definition='This technique is defined as "Making a large request first so a smaller one afterward seems more reasonable."',
        message_example=(
            "Would you consider signing up for a month-long smoking cessation retreat? If that's "
            "too big a step, maybe you'd try a nicotine patch for one week instead."
        ),
    ),
    # 10. Public Commitment
    PersuasionTechnique(
        technique_id=10,
        name="Public Commitment",
        strategy="Commitment-based",
        category="Ethical",
        definition='This technique is defined as "Having someone declare their position or intention publicly to strengthen follow-through."',
        message_example=(
            "Tell your friends and family you plan to quit—once you announce it, you’ll feel more "
            "motivated to stick with it."
        ),
    ),
    # 11. Alliance Building
    PersuasionTechnique(
        technique_id=11,
        name="Alliance Building",
        strategy="Relationship-based",
        category="Ethical",
        definition='This technique is defined as "Forming coalitions or partnerships to strengthen persuasion."',
        message_example=(
            "Let’s tackle this together—I’ll help you find resources, and we can keep each other "
            "accountable so you can quit smoking."
        ),
    ),
    # 12. Complimenting
    PersuasionTechnique(
        technique_id=12,
        name="Complimenting",
        strategy="Relationship-based",
        category="Ethical",
        definition='This technique is defined as "Using praise or flattery to foster goodwill and encourage compliance."',
        message_example=(
            "You’ve always shown such determination—put that strength toward quitting smoking, "
            "and I believe you’ll succeed."
        ),
    ),
    # 13. Shared Values
    PersuasionTechnique(
        technique_id=13,
        name="Shared Values",
        strategy="Relationship-based",
        category="Ethical",
        definition='This technique is defined as "Highlighting common beliefs or principles to create rapport."',
        message_example=(
            "We both care about living healthier, longer lives—quitting smoking aligns perfectly "
            "with that goal."
        ),
    ),
    # 14. Relationship Leverage
    PersuasionTechnique(
        technique_id=14,
        name="Relationship Leverage",
        strategy="Relationship-based",
        category="Ethical",
        definition='This technique is defined as "Drawing on personal or emotional bonds to influence behavior."',
        message_example=(
            "It would mean so much to me if you quit smoking—I care about you and want you to stay healthy."
        ),
    ),
    # 15. Loyalty Appeals
    PersuasionTechnique(
        technique_id=15,
        name="Loyalty Appeals",
        strategy="Relationship-based",
        category="Ethical",
        definition='This technique is defined as "Urging someone to act out of allegiance to a person, group, or cause."',
        message_example=(
            "Your family really needs you in the long run—quitting smoking shows your commitment "
            "to being there for them."
        ),
    ),
    # 16. Favor
    PersuasionTechnique(
        technique_id=16,
        name="Favor",
        strategy="Exchange-based",
        category="Ethical",
        definition='This technique is defined as "Offering or requesting a small courtesy or help that can facilitate further compliance."',
        message_example=(
            "If I help you find a free cessation program, would you consider trying it out "
            "as a personal favor to me?"
        ),
    ),
    # 17. Negotiation
    PersuasionTechnique(
        technique_id=17,
        name="Negotiation",
        strategy="Exchange-based",
        category="Ethical",
        definition='This technique is defined as "Exchanging concessions and proposals to reach a mutually acceptable agreement."',
        message_example=(
            "If you quit smoking, I'll make sure we create a positive smoke-free environment at home—"
            "how does that sound?"
        ),
    ),
    # 18. Encouragement
    PersuasionTechnique(
        technique_id=18,
        name="Encouragement",
        strategy="Appraisal-based",
        category="Ethical",
        definition='This technique is defined as "Providing supportive, positive messages to motivate action."',
        message_example=(
            "You have the strength to quit smoking—every step you take is progress toward a healthier life."
        ),
    ),
    # 19. Affirmation
    PersuasionTechnique(
        technique_id=19,
        name="Affirmation",
        strategy="Appraisal-based",
        category="Ethical",
        definition='This technique is defined as "Reinforcing self-worth or confidence to inspire desired behavior."',
        message_example=(
            "I know you’re strong and capable—you can absolutely overcome smoking if you put your mind to it."
        ),
    ),
    # 20. Positive Emotional Appeal
    PersuasionTechnique(
        technique_id=20,
        name="Positive Emotional Appeal",
        strategy="Emotion-based",
        category="Ethical",
        definition='This technique is defined as "Eliciting uplifting emotions (e.g., hope, joy) to influence."',
        message_example=(
            "Imagine the relief and pride you’ll feel once you’ve quit—it’s a wonderful gift to yourself."
        ),
    ),
    # 21. Negative Emotional Appeal
    PersuasionTechnique(
        technique_id=21,
        name="Negative Emotional Appeal",
        strategy="Emotion-based",
        category="Ethical",
        definition='This technique is defined as "Stirring distressing feelings (e.g., fear, guilt) to prompt action."',
        message_example=(
            "Consider the fear of future health complications—quitting now can spare you and your family that pain."
        ),
    ),
    # 22. Storytelling
    PersuasionTechnique(
        technique_id=22,
        name="Storytelling",
        strategy="Emotion-based",
        category="Ethical",
        definition='This technique is defined as "Using engaging narratives or personal tales to connect and persuade."',
        message_example=(
            "My grandfather smoked for 40 years; when he finally quit, he said it was the best choice he ever made—"
            "you can have that same transformative story."
        ),
    ),
    # 23. Anchoring
    PersuasionTechnique(
        technique_id=23,
        name="Anchoring",
        strategy="Information Bias",
        category="Ethical",
        definition='This technique is defined as "Setting an initial reference point that biases subsequent judgments."',
        message_example=(
            "If you think about spending $300 a month on cigarettes, quitting suddenly seems like "
            "an affordable way to save money."
        ),
    ),
    # 24. Priming
    PersuasionTechnique(
        technique_id=24,
        name="Priming",
        strategy="Information Bias",
        category="Ethical",
        definition='This technique is defined as "Activating certain ideas or memories that shape how new information is interpreted."',
        message_example=(
            "Before we talk about smoking, can you recall the last time you felt truly healthy and energetic?"
        ),
    ),
    # 25. Framing
    PersuasionTechnique(
        technique_id=25,
        name="Framing",
        strategy="Information Bias",
        category="Ethical",
        definition='This technique is defined as "Presenting information in a way that emphasizes specific aspects and influences perception."',
        message_example=(
            "Instead of focusing on losing something you enjoy, think of quitting smoking as "
            "gaining extra years of life."
        ),
    ),
    # 26. Confirmation Bias
    PersuasionTechnique(
        technique_id=26,
        name="Confirmation Bias",
        strategy="Information Bias",
        category="Ethical",
        definition='This technique is defined as "Steering people to focus only on evidence that supports their existing beliefs."',
        message_example=(
            "You’ve already noticed coughing and reduced stamina—quitting proves these symptoms "
            "really do come from smoking."
        ),
    ),
    # 27. Reciprocity
    PersuasionTechnique(
        technique_id=27,
        name="Reciprocity",
        strategy="Linguistics-based",
        category="Ethical",
        definition='This technique is defined as "Prompting a sense of obligation to return favors or kindnesses."',
        message_example=(
            "I’ve supported you through tough times; will you try quitting smoking as a personal favor to me?"
        ),
    ),
    # 28. Compensation
    PersuasionTechnique(
        technique_id=28,
        name="Compensation",
        strategy="Linguistics-based",
        category="Ethical",
        definition='This technique is defined as "Offering rewards, payment, or incentives in exchange for compliance."',
        message_example=(
            "If you manage to quit smoking for a month, I’ll treat you to that vacation you’ve been dreaming of."
        ),
    ),
    # 29. Supply Scarcity
    PersuasionTechnique(
        technique_id=29,
        name="Supply Scarcity",
        strategy="Scarcity-based",
        category="Ethical",
        definition='This technique is defined as "Suggesting limited availability to increase perceived value or urgency."',
        message_example=(
            "There’s only a short window of time before a new tax on cigarettes comes into effect—"
            "this is the perfect moment to quit."
        ),
    ),
    # 30. Time Pressure
    PersuasionTechnique(
        technique_id=30,
        name="Time Pressure",
        strategy="Scarcity-based",
        category="Ethical",
        definition='This technique is defined as "Imposing deadlines or urgency to spur quicker decisions."',
        message_example=(
            "Smoking cessation classes fill up quickly; you need to sign up now if you want to join "
            "this month’s program."
        ),
    ),
    # 31. Reflective Thinking
    PersuasionTechnique(
        technique_id=31,
        name="Reflective Thinking",
        strategy="Reflection-based",
        category="Ethical",
        definition='This technique is defined as "Encouraging deliberate self-examination or introspection before choosing."',
        message_example=(
            "Take a moment to consider how smoking aligns with your personal goals—where do you see "
            "it fitting into your future?"
        ),
    ),
    # 32. Threats
    PersuasionTechnique(
        technique_id=32,
        name="Threats",
        strategy="Threat",
        category="Unethical",
        definition='This technique is defined as "Using intimidation or the promise of harm to force compliance."',
        message_example=(
            "If you don’t quit smoking, you’ll be kicked out of the house—no exceptions."
        ),
    ),
    # 33. False Promises
    PersuasionTechnique(
        technique_id=33,
        name="False Promises",
        strategy="Deception",
        category="Unethical",
        definition='This technique is defined as "Making commitments with no intention or ability to fulfill them."',
        message_example=(
            "If you quit smoking today, I'll immediately send you on a fully paid world tour "
            "(despite having no real plan to do so)."
        ),
    ),
    # 34. Misrepresentation
    PersuasionTechnique(
        technique_id=34,
        name="Misrepresentation",
        strategy="Deception",
        category="Unethical",
        definition='This technique is defined as "Distorting or exaggerating facts to mislead."',
        message_example=(
            "I heard that smoking just one cigarette can give you an incurable disease instantly, "
            "so you must quit now."
        ),
    ),
    # 35. False Information
    PersuasionTechnique(
        technique_id=35,
        name="False Information",
        strategy="Deception",
        category="Unethical",
        definition='This technique is defined as "Disseminating outright untruths or fake data to deceive."',
        message_example=(
            "Quitting smoking guarantees you’ll never catch a cold again—so stop now."
        ),
    ),
    # 36. Rumors
    PersuasionTechnique(
        technique_id=36,
        name="Rumors",
        strategy="Social Sabotage",
        category="Unethical",
        definition='This technique is defined as "Spreading unverified, potentially damaging talk to manipulate perceptions."',
        message_example=(
            "People are saying you’re too weak to quit—wouldn’t you like to prove them wrong?"
        ),
    ),
    # 37. Social Punishment
    PersuasionTechnique(
        technique_id=37,
        name="Social Punishment",
        strategy="Social Sabotage",
        category="Unethical",
        definition='This technique is defined as "Threatening or applying social consequences for not conforming."',
        message_example=(
            "Keep smoking, and I’ll make sure all your friends stop inviting you out."
        ),
    ),
    # 38. Creating Dependency
    PersuasionTechnique(
        technique_id=38,
        name="Creating Dependency",
        strategy="Social Sabotage",
        category="Unethical",
        definition='This technique is defined as "Cultivating reliance or subservience to maintain control."',
        message_example=(
            "You can only stay in my program if you do exactly as I say, including quitting smoking when I demand."
        ),
    ),
    # 39. Exploiting Weakness
    PersuasionTechnique(
        technique_id=39,
        name="Exploiting Weakness",
        strategy="Social Sabotage",
        category="Unethical",
        definition='This technique is defined as "Targeting a person’s vulnerabilities or insecurities for manipulation."',
        message_example=(
            "You already struggle with anxiety, so I’ll keep reminding you that smoking makes your anxiety worse "
            "until you do what I want."
        ),
    ),
    # 40. Discouragement
    PersuasionTechnique(
        technique_id=40,
        name="Discouragement",
        strategy="Social Sabotage",
        category="Unethical",
        definition='This technique is defined as # "Undermining someone’s confidence or optimism to steer their behavior."',
        message_example=(
            "You’re never going to successfully quit anyway, so you might as well do what I say "
            "if you even try."
        ),
    ),
]


def sample_technique(
    n: int = 1, only_ethical: bool = False, only_unethical: bool = False
) -> List[PersuasionTechnique]:
    """
    Returns a random sample of persuasion techniques.

    :param n: How many techniques to sample (default=1).
    :param only_ethical: If True, sample only from ethical techniques.
    :param only_unethical: If True, sample only from unethical techniques.
    :return: A list of PersuasionTechnique objects.
    """
    # Narrow down the pool based on category request
    if only_ethical and not only_unethical:
        pool = [t for t in ALL_TECHNIQUES if t.category.lower() == "ethical"]
    elif only_unethical and not only_ethical:
        pool = [t for t in ALL_TECHNIQUES if t.category.lower() == "unethical"]
    else:
        pool = ALL_TECHNIQUES  # no filtering

    # Just return a random sample of size n (or fewer if the pool is smaller)
    return random.sample(pool, k=min(n, len(pool)))
