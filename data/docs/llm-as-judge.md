# Using a language model as a judge

The LLM-as-judge pattern uses a language model to grade the output of another
model, or of itself. A judge prompt supplies the question, the context and the
candidate answer, and asks for a score and a short justification, often in a
structured format such as JSON.

Judges are cheap and scale well, but they have known biases. A model grading
its own output tends to be lenient toward it, a bias called self-preference.
Judges can favour longer answers, can be swayed by confident wording, and can
produce inconsistent scores for the same input. Small models are noticeably
less reliable judges than large ones.

Good practice is to keep the judge's task narrow and concrete, for example
asking whether each claim is supported by the context rather than asking for
an overall quality rating. Running the judge at temperature 0 makes it more
repeatable. Requesting an explanation alongside the score makes failures
easier to diagnose, and in a self-correcting system that explanation can be
passed back to the generator so the next attempt knows what to fix.

Judge output must be parsed defensively. Models do not always follow format
instructions and may write a score as a decimal, a percentage such as 85%, or
a fraction such as 8/10. When no score can be extracted, or the judge cannot
be reached, the result should be recorded as unavailable rather than as a low
score, because a missing judgement says nothing about answer quality.
