"""
The video script: one entry per slide.

Fields:
    title:     slide header
    bullets:   on-screen bullet points (kept short — this is a video, not a slide deck)
    diagram:   optional filename (without extension) from diagrams/ to display
    narration: the text spoken by the TTS voice-over (this is the real teaching content)
"""

SLIDES = [
    {
        "title": "Just Enough Calculus for Machine Learning",
        "bullets": ["A beginner's guide — no prior calculus or linear algebra assumed"],
        "diagram": None,
        "narration": (
            "Welcome. This video teaches exactly the calculus you need to understand how "
            "machine learning models learn — nothing more, nothing less. We start from zero. "
            "If you remember basic algebra, you're ready."
        ),
    },
    {
        "title": "Why Does Machine Learning Need Calculus?",
        "bullets": [
            "A model has adjustable numbers, called parameters",
            "Training means slowly changing those numbers to make fewer mistakes",
            "Calculus tells us exactly which direction, and how much, to change each one",
        ],
        "diagram": None,
        "narration": (
            "Every machine learning model has a set of adjustable numbers, called parameters. "
            "Training a model means slowly changing those numbers so the model makes fewer "
            "and fewer mistakes. But how do you know which direction to change a number, and "
            "by how much? That is exactly the question calculus answers. That is the entire "
            "reason this video exists."
        ),
    },
    {
        "title": "Quick Refresher: What Is a Function?",
        "bullets": [
            "A function is a rule: put a number in, get a number out",
            "Written f(x) — 'f of x'",
            "Example: f(x) = x squared. f(3) = 9",
        ],
        "diagram": None,
        "narration": (
            "Before we talk about calculus, let's make sure function notation feels comfortable. "
            "A function is simply a rule that takes a number in, and produces a number out. We "
            "write it as f of x. For example, if f of x equals x squared, then f of 3 equals 9, "
            "because 3 times 3 is 9. That's all a function is — an input, output machine."
        ),
    },
    {
        "title": "The Big Idea: Rate of Change",
        "bullets": [
            "Speed = change in distance ÷ change in time",
            "This ratio is a RATE OF CHANGE",
            "Calculus is the mathematics of rates of change, for ANY function",
        ],
        "diagram": None,
        "narration": (
            "Here is the single biggest idea in this whole video. Think about speed. Speed is "
            "just the change in distance divided by the change in time. That ratio is called a "
            "rate of change. Calculus takes this everyday idea — a rate of change — and makes it "
            "precise enough to apply to any function at all, not just distance and time."
        ),
    },
    {
        "title": "The Slope of a Straight Line",
        "bullets": [
            "Slope = rise ÷ run = change in y ÷ change in x",
            "For a straight line, the slope is the SAME everywhere",
        ],
        "diagram": "slope_line",
        "narration": (
            "You've likely seen this before: the slope of a straight line is rise over run — how "
            "much y changes, divided by how much x changes. In this picture, the line rises 3 "
            "units while running 2 units, so the slope is 3 divided by 2, or 1.5. For a straight "
            "line, this slope is exactly the same no matter where you measure it."
        ),
    },
    {
        "title": "But Curves Don't Have Just ONE Slope",
        "bullets": [
            "A curve's steepness changes at every point",
            "We need the slope AT one specific point — not for the whole curve",
        ],
        "diagram": "varying_slope",
        "narration": (
            "A straight line has one constant slope. But most of the functions we care about in "
            "machine learning are curves, and a curve's steepness is different at every point. "
            "Look at this parabola: it's steep and negative on the left, flat at the bottom, and "
            "steep and positive on the right. We need a way to talk about the slope at one exact "
            "point. That's exactly what a derivative gives us."
        ),
    },
    {
        "title": "From Secant Line to Tangent Line",
        "bullets": [
            "Draw a line through two nearby points on the curve — a secant line",
            "Slide the second point closer and closer to the first",
            "The secant line approaches the TANGENT line — touching just one point",
        ],
        "diagram": "secant_to_tangent",
        "narration": (
            "Here's how we find the slope at one exact point. First, pick two points on the curve "
            "and draw a line through both of them — this is called a secant line. Now slide the "
            "second point closer and closer to the first. Watch what happens to the dashed lines "
            "in this picture: as the gap, called h, shrinks toward zero, the secant line rotates "
            "and settles into the tangent line — the line that just grazes the curve at a single "
            "point. That limiting slope is the derivative."
        ),
    },
    {
        "title": "The Derivative — A Formal Definition",
        "bullets": [
            "f'(x) = the limit, as h approaches 0, of  [f(x+h) − f(x)] ÷ h",
            "This is just the secant slope, with the gap h shrunk to zero",
        ],
        "diagram": None,
        "narration": (
            "Now let's write down what we just saw, in symbols. The derivative of f at x, written "
            "f prime of x, is defined as the limit, as h approaches zero, of f of x plus h, minus "
            "f of x, all divided by h. Don't let the symbols intimidate you — that fraction is "
            "exactly the secant line's slope from the last slide. We are simply asking: what does "
            "that slope approach as the gap h shrinks all the way to zero?"
        ),
    },
    {
        "title": "Reading Derivative Notation",
        "bullets": [
            "f'(x)  —  read: 'f prime of x'",
            "dy/dx  —  read: 'the derivative of y with respect to x'",
            "All of these mean exactly the same thing",
        ],
        "diagram": None,
        "narration": (
            "You will see the derivative written a few different ways, and they all mean exactly "
            "the same thing. F prime of x, with a small apostrophe. D y d x, which reads as the "
            "derivative of y with respect to x. And sometimes d over d x of f of x. Different "
            "textbooks, same idea: the exact slope of the function at a point."
        ),
    },
    {
        "title": "Derivative Rule 1: The Power Rule",
        "bullets": [
            "d/dx[x^n] = n · x^(n−1)",
            "Example: d/dx[x^2] = 2x",
            "Example: d/dx[x^3] = 3x^2",
        ],
        "diagram": None,
        "narration": (
            "Here is the single most useful shortcut in this entire video: the power rule. To "
            "differentiate x to the power of n, bring the exponent down in front, and reduce the "
            "exponent by one. So the derivative of x squared is 2 x. The derivative of x cubed is "
            "3 x squared. You almost never need the limit definition directly — this shortcut "
            "handles the vast majority of cases you'll meet."
        ),
    },
    {
        "title": "Derivative Rule 2: Constants and Sums",
        "bullets": [
            "The derivative of a constant is 0 — constants don't change",
            "The derivative of a sum is the sum of the derivatives",
            "Example: d/dx[3x^2 + 5] = 6x + 0 = 6x",
        ],
        "diagram": None,
        "narration": (
            "Two more simple rules. First, the derivative of a plain constant, like 5, is zero — "
            "a constant never changes, so its rate of change is nothing. Second, if you're "
            "differentiating a sum of terms, you can just differentiate each term separately and "
            "add the results. So the derivative of 3 x squared plus 5 is 6 x plus zero, which is "
            "just 6 x."
        ),
    },
    {
        "title": "Derivatives You'll See Everywhere in ML",
        "bullets": [
            "d/dx[e^x] = e^x  — its own derivative!",
            "d/dx[ln(x)] = 1/x",
            "These two appear in almost every loss function you'll ever use",
        ],
        "diagram": None,
        "narration": (
            "Two special functions deserve their own slide, because they appear constantly in "
            "machine learning. The exponential function, e to the x, has a remarkable property: "
            "its own derivative is itself, e to the x. And the natural logarithm, ell n of x, has "
            "derivative 1 over x. Keep these two in your back pocket — you will meet them again "
            "and again in loss functions and probability models."
        ),
    },
    {
        "title": "The Chain Rule — The Most Important Rule for ML",
        "bullets": [
            "For a composed function f(g(x)), the derivative is f'(g(x)) · g'(x)",
            "Neural networks are long CHAINS of composed functions",
            "This is literally how backpropagation computes gradients",
        ],
        "diagram": None,
        "narration": (
            "This next rule is the single most important one for machine learning: the chain "
            "rule. When one function is nested inside another — f of g of x — its derivative is "
            "the derivative of the outer function, evaluated at the inner function, multiplied by "
            "the derivative of the inner function. Why does this matter so much? Because a neural "
            "network is nothing more than a long chain of composed functions, layer after layer. "
            "Backpropagation, the algorithm that trains neural networks, is the chain rule applied "
            "mechanically, layer by layer, from the output back to the input."
        ),
    },
    {
        "title": "Partial Derivatives — More Than One Input",
        "bullets": [
            "Real models have MANY parameters, not just one",
            "A partial derivative asks: how does the output change if I nudge ONE parameter,",
            "holding all the others fixed?  Notation: ∂f/∂x",
        ],
        "diagram": None,
        "narration": (
            "So far we've only dealt with functions of one variable. But a real machine learning "
            "model might have millions of parameters. A partial derivative asks a very focused "
            "question: if I nudge just ONE of those parameters a tiny bit, holding every other "
            "parameter exactly fixed, how does the output change? We write this with a curly d "
            "symbol: partial f, partial x. It's computed exactly like an ordinary derivative — you "
            "just treat every other variable as if it were a constant."
        ),
    },
    {
        "title": "The Gradient — A Compass Pointing Uphill",
        "bullets": [
            "Stack ALL the partial derivatives into one vector",
            "That vector is called the GRADIENT",
            "The gradient points in the direction of STEEPEST INCREASE",
        ],
        "diagram": None,
        "narration": (
            "Now take every one of those partial derivatives — one for each parameter — and stack "
            "them into a single vector. That vector is called the gradient. The gradient has a "
            "beautiful geometric meaning: it always points in the direction where the function "
            "increases the fastest — like a compass that always points uphill, no matter where you "
            "stand on the surface."
        ),
    },
    {
        "title": "Gradient Descent — Walking Downhill to Learn",
        "bullets": [
            "To REDUCE error, step in the OPPOSITE direction of the gradient",
            "Repeat this update, over and over",
            "This simple loop is literally how models learn",
        ],
        "diagram": "gradient_descent",
        "narration": (
            "Since the gradient points uphill, and we want to REDUCE our model's error, we simply "
            "step in the exact opposite direction — downhill. Compute the gradient, take a small "
            "step backward along it, and repeat. Watch the numbered dots in this picture roll down "
            "the bowl toward the bottom, step by step. This simple repeating loop, called gradient "
            "descent, is the mechanism behind almost every model training process in modern machine "
            "learning."
        ),
    },
    {
        "title": "Putting It All Together: A Tiny Example",
        "bullets": [
            "Fit y = w·x to the point (x=2, y=6), starting from w=1",
            "Error = (prediction − actual)^2 = (w·2 − 6)^2",
            "Gradient with respect to w: 2·(w·2−6)·2 = at w=1, gradient = −16",
            "Step opposite the gradient: w increases, prediction improves",
        ],
        "diagram": None,
        "narration": (
            "Let's tie everything together with one tiny worked example. Suppose we're fitting a "
            "line, y equals w times x, to a single data point where x is 2 and the true y is 6. We "
            "start with a guess, w equals 1. Our prediction is w times 2, which is 2 — far from 6. "
            "The squared error is w times 2, minus 6, squared. Using the power rule and the chain "
            "rule together, the derivative of this error with respect to w works out to 2 times, w "
            "times 2 minus 6, times 2. Plugging in w equals 1, that gradient is negative 16. Since "
            "the gradient is negative, gradient descent tells us to increase w — and sure enough, "
            "increasing w moves our prediction closer to 6. That's calculus, doing exactly its job."
        ),
    },
    {
        "title": "Exercises — Practice What You Learned",
        "bullets": [
            "1. Compute d/dx[x^4]  and  d/dx[7x^3 + 2x]",
            "2. Compute d/dx[e^x]  and  d/dx[ln(x)]  from memory",
            "3. For f(x)=(3x+1)^2, use the chain rule to find f'(x)",
            "4. For f(x,y)=x^2·y, find ∂f/∂x and ∂f/∂y",
            "5. Explain, in your own words, why gradient descent moves OPPOSITE the gradient",
            "6. Given gradient = −16 and learning rate 0.1, compute the new value of w starting from w=1",
        ],
        "diagram": None,
        "narration": (
            "Now it's your turn. Pause the video and work through these six exercises. Number one: "
            "compute the derivative of x to the fourth, and of 7 x cubed plus 2 x, using the power "
            "rule. Number two: write down the derivatives of e to the x and natural log of x, from "
            "memory. Number three: use the chain rule to differentiate the quantity 3 x plus 1, "
            "squared. Number four: find both partial derivatives of x squared times y. Number five: "
            "explain, in your own words, why gradient descent steps opposite the gradient rather than "
            "along it. And number six: given a gradient of negative 16, a learning rate of 0.1, and a "
            "starting value of w equals 1, compute the updated value of w after one gradient descent "
            "step."
        ),
    },
    {
        "title": "Summary — What You Now Know",
        "bullets": [
            "A derivative is the exact slope of a function at one point",
            "The power rule, and the derivatives of e^x and ln(x)",
            "The chain rule — the engine behind backpropagation",
            "The gradient points uphill; gradient descent walks downhill to learn",
        ],
        "diagram": None,
        "narration": (
            "Let's recap. A derivative is the exact slope of a function at a single point, found by "
            "shrinking a secant line down to a tangent line. The power rule handles most ordinary "
            "derivatives, while e to the x and natural log of x have their own special forms. The "
            "chain rule lets us differentiate through composed functions — and it is exactly the "
            "mechanism behind backpropagation in neural networks. And the gradient, a vector of "
            "partial derivatives, always points uphill — so gradient descent simply walks the "
            "opposite way, step after step, until the model's error is as small as we can make it. "
            "That is genuinely just enough calculus to understand how machine learning learns. "
            "Thanks for watching."
        ),
    },
]
