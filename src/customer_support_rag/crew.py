from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class CustomerSupportCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def query_agent(self) -> Agent:
        return Agent(config=self.agents_config["query_agent"], verbose=True)

    @agent
    def retrieval_agent(self) -> Agent:
        return Agent(config=self.agents_config["retrieval_agent"], verbose=True)

    @agent
    def response_agent(self) -> Agent:
        return Agent(config=self.agents_config["response_agent"], verbose=True)

    @agent
    def escalation_agent(self) -> Agent:
        return Agent(config=self.agents_config["escalation_agent"], verbose=True)

    @task
    def query_task(self) -> Task:
        return Task(config=self.tasks_config["query_task"])

    @task
    def retrieval_task(self) -> Task:
        return Task(config=self.tasks_config["retrieval_task"])

    @task
    def response_task(self) -> Task:
        return Task(
            config=self.tasks_config["response_task"],
            output_file="blobs/answer.md"
        )

    @task
    def escalation_task(self) -> Task:
        return Task(config=self.tasks_config["escalation_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
