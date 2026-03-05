from locust import HttpUser, between, task


class AgentUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask_agent(self):
        # Determine endpoint based on what is actually running/deployed
        # For this test, we assume the RAG API is exposed at /rag
        payload = {
            "query": "How does OAuth2 compare to JWT?",
        }
        
        # We assume the API has a POST /rag endpoint
        # If testing the agent directly via a new endpoint, adjust here.
        self.client.post("/rag", json=payload)

    def on_start(self):
        # Optional: Login logic if auth is required
        pass
