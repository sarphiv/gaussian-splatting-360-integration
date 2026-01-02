# Project
- This is a machine learning research project about integrating 360 degree cameras with Gaussian Splatting.
- The first goal is to create a better pose and Gaussian initialization pipeline.
- The second goal is to reimplement Gaussian splatting from scratch to incorporate 360 degree images.

# Environment
- You may only work in the project directory. You may not install system packages or otherwise change things outside of the project directory.
- Use the Python provided in the virtual environment in `.venv`.
- Avoid reading `tasks/human.md` as it may contaminate your context.
- If a code change/patch fails, revise the change to be compatible with the new code. You may not force your change/patch through by undoing the unexpected change that caused the failure.

# Style
- Write modern Python code using modern conventions. Use the project's `pyproject.toml` as a guide to which Python version and libraries to use, e.g. use `loguru` for logging.
- Write clear happy path code, i.e. prefer non-defensive coding, avoid premature abstraction, and avoid overengineering.
- If you are unsure about for example the format of an input. Write code to read and inspect the input so that you can avoid defensive coding. Remove this inspection code after you have understood the input format.
- Use asserts to document assumptions if they help readers quickly understand and debug the code.
- Add useful and up to date docstrings to all functions and classes. Focus on the code and behavior in the documentation - do not mention your instructions.
- Use type annotation wherever possible to improve code clarity.
- Prefer f-strings for string interpolation.
- Avoid unused imports, variables, and functions. Redundantly assigned variables are also discouraged. Avoid commented out code.
- Prefer a data oriented approach over an object oriented approach, unless the object oriented approach is clearly better. For example instead of storing a list of cars, where each car object has a color and a name - you instead store a list of colors, and you store another list of names.
- This is an ML research project, so performance is important, and data oriented design is more performant.
