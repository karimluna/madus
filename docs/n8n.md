## How-to guide for n8n integration

n8n acts as the __integration and ingestion__ layer for __MADUS__. Instead of users having to write `curl` commands, creating a frontend or using the FastAPI Swagger UI to interact with the backend, n8n can be connected to the AI pipeline directly to everyday tools (file folders, email, Slack, Google Drive) with __zero code__.


### Core workflow
The workflow is a 3-step pipeline configured in the n8n browser UI:

1. __Triggern Node (Watch Folder)__: n8n monitors a specific local directory or cloud storage bucket (like Drive or S3). When a new PDF is dropped there it triggers the workflow.

2. __Action Node (HTTP Request)__: n8n takes that PDF file and makes a `multipart/form-data` POST request to the `MADUS` API. It passes the file and a default question of course.

3. __Output Node (Slack / Google Sheets)__: When the API returns the `final_answer` and `confidence` score, n8n routes that JSON response to a readable destination, posting a messsage to a Slack channel, adding a row to a Google Sheet, or sending an email, this is completely personalizable.

This enforces a clean separation of concerns, wiring an HTTP request, a file watcher and a Slack webhook in a browser UI takes 5 minutes this in Python would take hours. Also it provides a lot of __flexibility__, changing matters is just about changing drag-drop nodes in n8n. This means that `app.py` is a stateless JSON API so it does not need to know _where_ the PDF came from or where the answer is going.



### How questiond gets passes
Threr are two common patterns to handle this in n8n:

__Hardcoded__: n8n HTTP Request node always sends a generic question like "What is the core idea and the results of the paper?", turning the system into an automatic document summarizer.

__Dynamic__ (Filename convention)__: This pattern typical enforces a filename convention like `what_is_the_revenue_Q3.pdf` and n8n uses an "Expression" to replace underscores with spaces, strips the `.pdf` and passes that string as `question` in form data.



### Finally how to set it up in a local environment
When you run `docker compose up`, n8n starts on port `5678`. Here is how the configure the magic "Drop PDF → Slack":

1. Open `http://localhost:5678` and log in (admin / madus).
2. Click **Add Workflow**.
3. Add a **Trigger** node (e.g., "Read Binary File from Folder" or a "Webhook" if testing via Postman).
4. Add an **HTTP Request** node:
   * Method: `POST`
   * URL: `http://api:8000/api/analyze` (using Docker's internal DNS)
   * Body Content Type: `Multipart-Form Data`
   * Send Body: True
   * Add a file parameter named `file` pointing to the binary data from the trigger.
   * Add a string parameter named `question` with your query.
5. Add a **Slack** node:
   * Connect your Slack workspace.
   * Channel: `#madus-results`
   * Text: `Answer: {{$json.final_answer}} (Confidence: {{$json.confidence}})`

That is, the AI reasoning layer is abstracted away from the integration layer.