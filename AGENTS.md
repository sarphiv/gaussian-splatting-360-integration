# Project
- This is a machine learning research project about integrating 360 degree cameras with Gaussian Splatting.
- The first goal is to create a better pose and Gaussian initialization pipeline.
- The second goal is to reimplement Gaussian splatting from scratch to incorporate 360 degree images.


# Strategy
- Figure out what type of agent you are, and then follow the relevant instructions below.

- Agent types
    - If you have been directed to a specific task file, you are a subagent.
    - If you have been directed to no task file or `main.md`, you are an orchestrator agent.

- Orchestrator task initialization if there is no `main.md` file.
    - Create a `tasks/{task-name}/main.md` file. Copy `tasks/template` as a template.
    - Fill out the task file according to the task template.
    - If I approve the plan, proceed to execute it.
    - If the plan is rejected, update the plan according to feedback and resubmit for approval.
    - Once the plan is approved, you will need no further approval for changes to the plan unless they are major.
- Orchestration notes
    - The `main.md` task file focuses on carving out and delegating subtasks to subagents.
    - Place the subtask file at `tasks/{task-name}/{subtask-name}.md` and fill them out accordingly.
    - The subtasks should be designed to run in parallel while minimizing risk of conflict between subagents.
    - Each new subtask is tied to one new subagent.
    - You work in the same loop of general plan, detailed plan, implement, review, repeat/complete.
    - Except, your focus is broader and you spawn new subagents to do the implementation and review steps.
    - Reviewing a subtask is a subtask itself, so spawn a new subagent and prepare a filled out subtask for it.
    - Once a subtask is complete, read its task file, and use this to inform your next steps.
    - Update existing subtask files or create new ones as necessary.
    - Do not read other task directories in the `tasks/` directory.
- Implementer notes
    - Subagents may be running in parallel so assume files may have changed since you last read them.
    - Review your own code before submitting it for review.
- Reviewer notes
    - Avoid reading the subtask file of the subagent you are reviewing.
    - Ensure that all changes follow the style guide below.
    - Use web search if necessary to verify correctness of code.


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
- Prefer a data oriented approach over an object oriented approach, unless the object oriented approach is clearly better. For example instead of storing a list of cars, where each car object has a color and a name - you instead store a list of colors, and you store another list of names.
- This is an ML research project, so performance is important, and data oriented design is more performant.
