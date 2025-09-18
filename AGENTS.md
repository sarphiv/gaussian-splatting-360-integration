# Project
- This is a machine learning research project about integrating 360 degree cameras with Gaussian Splatting.
- The first goal is to create a better pose and Gaussian initialization pipeline.
- The second goal is to reimplement Gaussian splatting from scratch to incorporate 360 degree images.

# Strategy
- Create a `tasks/{task-name}.md` file for the task you are assigned.
- Plan out your changes first in general terms. Write this to the file.
- Extend each general change into a detailed plan in a new section of the file.
- Revise the plan if issues are identified and repeat the above process.
- Finally, write a detailed todo list at the bottom of the file together with any questions/assumptions you have.
- If I approve the plan, proceed to implement it, while continuing to keep the task file updated and revise it if necessary.
- You must always ask for approval for the very first initial plan. Later revisions to the plan do not need approval unless they deviate significantly from the original plan.
- After implementation, read through your changes and double check that they follow the style guide.

# Environment
- You may only work in the project directory. You may not install system packages or otherwise change things outside of the project directory.
- Use the Python provided in the virtual environment in `.venv`.
- Avoid reading `tasks/human.md` as it may contaminate your context.

# Style
- Write modern Python code using modern conventions. Use the project's `pyproject.toml` as a guide to which Python version and libraries to use, e.g. use `loguru` for logging.
- Write clear happy path code, i.e. prefer non-defensive coding, avoid premature abstraction, and avoid overengineering.
- If you are unsure about for example the format of an input. Write code to read and inspect the input so that you can avoid defensive coding. Remove this inspection code after you have understood the input format.
- Use asserts to document assumptions if they help readers quickly understand and debug the code.
- Add useful and up to date docstrings to all functions and classes. Focus on the code and behavior in the documentation - do not mention your instructions.
- Use type annotation wherever possible to improve code clarity.
- Avoid unused imports, variables, and functions. Redundantly assigned variables are also discouraged. Avoid commented out code. 
- Prefer a data oriented approach over an object oriented approach, unless the object oriented approach is clearly better. This is an ML research project, so performance is important. For example instead of storing a list of cars, where each car object has a color and a name - you instead store a list of colors, and you store another list of names. Data oriented design is more performant.