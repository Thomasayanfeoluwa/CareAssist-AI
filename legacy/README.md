# Legacy implementations — NOT deployed

## flask/app.py
Original single-user Flask app. Known defects (per project verdict):
shared module-level conversation memory, debug=True on 0.0.0.0,
chat history printed to logs, unsanitized marked.parse + innerHTML.

## streamlit/dashboard.py
Original Streamlit prototype. Uses st.cache_resource for a shared memory
object — not safe for multi-user use. Also calls ChatGroq directly.

## store_index.py, src/, templates/
Original indexing script, helper/prompt modules, and Flask HTML template.
Logic is ported into backend/app/ during Stage 1+.

None of this is imported, built, tested, or deployed.
