from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from google import generativeai as genai
from dotenv import load_dotenv
import os, re, traceback
from urllib.parse import unquote

# Try to import MySQL connector
try:
    import pymysql as MySQLdb
    MySQLdb.install_as_MySQLdb()
except ImportError:
    try:
        import MySQLdb
    except ImportError:
        print("ERROR: Please install pymysql: pip install pymysql")
        exit(1)

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
CORS(app)

# Global cache for database schema and intelligent query mapping
schema_cache = {}
query_cache = {}
table_mapping_cache = {}

class DynamicSQLAssistant:
    def __init__(self, db_creds):
        self.db_creds = db_creds
        self.connection = None
        self.cache_key = f"{self.db_creds.get('host')}_{self.db_creds.get('db')}"
        # Clear relevant caches when creating new assistant instance
        self.clear_cache()
        
    def clear_cache(self):
        """Clear all caches to ensure fresh data"""
        global schema_cache, query_cache, table_mapping_cache
        
        # Clear schema cache for this specific database
        if self.cache_key in schema_cache:
            del schema_cache[self.cache_key]
            
        # Clear all caches completely
        schema_cache.clear()
        query_cache.clear()
        table_mapping_cache.clear()
        
        print(f"All caches cleared for database: {self.cache_key}")
    
    def get_connection(self):
        """Get database connection with proper cleanup"""
        try:
            # Always close existing connection first
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
                
            self.connection = MySQLdb.connect(
                host=self.db_creds.get("host"),
                user=self.db_creds.get("user"),
                passwd=unquote(self.db_creds.get("pass", "")).strip(),
                db=self.db_creds.get("db"),
                charset="utf8mb4"
            )
            print(f"Fresh connection established to {self.db_creds.get('db')}")
            return self.connection
        except Exception as e:
            print(f"Connection error: {e}")
            raise e
    
    def discover_database_schema(self, force_refresh=False):
        """Dynamically discover and analyze the entire database schema"""
        
        # ALWAYS force refresh if explicitly requested or if no cache exists
        if force_refresh:
            print("Force refresh requested - clearing cache")
            if self.cache_key in schema_cache:
                del schema_cache[self.cache_key]
        
        # Check cache AFTER potential clearing
        if not force_refresh and self.cache_key in schema_cache:
            print("Using cached schema")
            return schema_cache[self.cache_key]
        
        print("Discovering fresh schema...")
        
        # Get a fresh connection
        conn = self.get_connection()
        cursor = conn.cursor()
        
        schema_info = {
            "tables": {},
            "relationships": [],
            "keywords": {},
            "discovery_timestamp": os.time.time() if hasattr(os, 'time') else 0
        }
        
        try:
            # Get all tables with better error handling
            print("Executing SHOW TABLES query...")
            cursor.execute("SHOW TABLES")
            raw_tables = cursor.fetchall()
            print(f"Raw tables from SHOW TABLES: {raw_tables}")
            
            if not raw_tables:
                print("WARNING: No tables found in database!")
                # Don't cache empty results - this might be a connection issue
                return schema_info
            
            tables = [table[0] for table in raw_tables]
            print(f"Processed tables: {tables}")
            
            # Process each table
            for table in tables:
                try:
                    print(f"Processing table: {table}")
                    
                    # Get table structure
                    cursor.execute(f"DESCRIBE `{table}`")
                    columns = cursor.fetchall()
                    print(f"  - Found {len(columns)} columns")
                    
                    # Get sample data with error handling
                    try:
                        cursor.execute(f"SELECT * FROM `{table}` LIMIT 5")
                        sample_data = cursor.fetchall()
                        print(f"  - Got {len(sample_data)} sample rows")
                    except Exception as e:
                        print(f"  - Error getting sample data from {table}: {e}")
                        sample_data = []
                    
                    # Get row count with error handling
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                        row_count = cursor.fetchone()[0]
                        print(f"  - Row count: {row_count}")
                    except Exception as e:
                        print(f"  - Error getting row count from {table}: {e}")
                        row_count = 0
                    
                    schema_info["tables"][table] = {
                        "columns": [{"name": col[0], "type": col[1], "null": col[2], "key": col[3]} for col in columns],
                        "sample_data": sample_data,
                        "row_count": row_count,
                        "column_names": [col[0].lower() for col in columns]
                    }
                    
                    # Build keyword mapping for intelligent query routing
                    table_keywords = [table.lower()]
                    for col in columns:
                        table_keywords.append(col[0].lower())
                    
                    schema_info["keywords"][table] = table_keywords
                    
                except Exception as e:
                    print(f"ERROR processing table {table}: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    continue
        
        except Exception as e:
            print(f"CRITICAL ERROR during schema discovery: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise e
        finally:
            try:
                cursor.close()
            except:
                pass
        
        # Only cache if we actually found tables
        if schema_info["tables"]:
            schema_cache[self.cache_key] = schema_info
            print(f"Schema cached for {len(schema_info['tables'])} tables")
        else:
            print("WARNING: No tables found - not caching empty schema")
        
        return schema_info
    
    def intelligent_table_detection(self, user_query):
        """Intelligently detect which table(s) the user is asking about"""
        schema = self.discover_database_schema()
        query_lower = user_query.lower()
        
        # Score tables based on keyword matches
        table_scores = {}
        
        for table_name, keywords in schema["keywords"].items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Exact match gets higher score
                    if keyword == query_lower.strip():
                        score += 10
                    # Partial match
                    elif keyword in query_lower.split():
                        score += 5
                    # Substring match
                    else:
                        score += 1
            table_scores[table_name] = score
        
        # Return tables sorted by relevance score
        relevant_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        return [table for table, score in relevant_tables if score > 0]
    
    def generate_intelligent_sql(self, user_query):
        """Generate SQL based on natural language understanding of the database"""
        schema = self.discover_database_schema()
        relevant_tables = self.intelligent_table_detection(user_query)
        
        if not relevant_tables:
            return None, "I couldn't find any relevant tables for your query. Available tables: " + ", ".join(schema["tables"].keys())
        
        primary_table = relevant_tables[0]
        table_info = schema["tables"][primary_table]
        
        # Create comprehensive context for AI
        context = f"""
DATABASE SCHEMA ANALYSIS:
Primary Table: `{primary_table}` ({table_info['row_count']} rows)
Columns: {', '.join([f"{col['name']} ({col['type']})" for col in table_info['columns']])}

Available Tables: {', '.join(schema['tables'].keys())}

Sample Data from {primary_table}:
{table_info['sample_data'][:3] if table_info['sample_data'] else 'No sample data'}

USER QUERY: {user_query}

Generate a MySQL SELECT query that answers the user's question. 
- Use proper MySQL syntax
- Consider the actual data types and structure
- If the query seems to want data from multiple tables, suggest JOIN operations
- For aggregations, use appropriate GROUP BY clauses
- Return only the SQL query, no explanations
"""
        
        try:
            response = model.generate_content(context)
            sql_query = re.sub(r"^```[\w]*\s*|```$", "", response.text.strip(), flags=re.IGNORECASE).strip()
            
            return sql_query, None
        except Exception as e:
            return None, f"Error generating SQL: {str(e)}"
    
    def execute_dynamic_query(self, user_query):
        """Execute a natural language query dynamically"""
        # Generate fresh SQL for each query to avoid stale cached queries
        sql_query, error = self.generate_intelligent_sql(user_query)
        if error:
            return {"error": error}
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            
            return {
                "query": sql_query,
                "results": [dict(zip(columns, row)) for row in results],
                "total_results": len(results),
                "columns": columns
            }
        except Exception as e:
            return {"error": f"Query execution failed: {str(e)}", "sql": sql_query}

# Global assistant instance
assistant = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/connect", methods=["POST"])
def connect_database():
    """Connect and analyze database schema"""
    global assistant
    try:
        creds = request.get_json()
        
        # Create new assistant instance (this will clear caches)
        assistant = DynamicSQLAssistant(creds)
        
        # Test connection and discover schema with FORCE refresh
        print("FORCING SCHEMA REFRESH on connect...")
        schema = assistant.discover_database_schema(force_refresh=True)
        
        return jsonify({
            "status": "Connected successfully",
            "database": creds.get("db"),
            "tables_found": len(schema["tables"]),
            "tables": list(schema["tables"].keys()),
            "total_records": sum([info["row_count"] for info in schema["tables"].values()])
        })
    except Exception as e:
        print(f"Connection error: {traceback.format_exc()}")
        return jsonify({"error": f"Connection failed: {str(e)}"})

@app.route("/query", methods=["POST"])
def dynamic_query():
    """Handle natural language queries dynamically"""
    global assistant
    
    if not assistant:
        return jsonify({"error": "Please connect to database first"})
    
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        
        if not user_query:
            return jsonify({"error": "Please provide a query"})
        
        result = assistant.execute_dynamic_query(user_query)
        return jsonify(result)
        
    except Exception as e:
        print(f"Query error: {traceback.format_exc()}")
        return jsonify({"error": f"Query processing failed: {str(e)}"})

@app.route("/schema", methods=["POST"])
def get_schema():
    """Get complete database schema information"""
    global assistant
    
    if not assistant:
        return jsonify({"error": "Please connect to database first"})
    
    try:
        # ALWAYS force refresh schema to get latest data
        print("FORCING SCHEMA REFRESH on schema request...")
        schema = assistant.discover_database_schema(force_refresh=True)
        return jsonify(schema)
    except Exception as e:
        print(f"Schema error: {traceback.format_exc()}")
        return jsonify({"error": f"Schema analysis failed: {str(e)}"})

@app.route("/refresh", methods=["POST"])
def refresh_schema():
    """Manually refresh database schema"""
    global assistant
    
    if not assistant:
        return jsonify({"error": "Please connect to database first"})
    
    try:
        # Clear cache and refresh
        print("MANUAL REFRESH REQUESTED - clearing all caches...")
        assistant.clear_cache()
        
        # Force refresh with new connection
        schema = assistant.discover_database_schema(force_refresh=True)
        
        return jsonify({
            "status": "Schema refreshed successfully",
            "tables_found": len(schema["tables"]),
            "tables": list(schema["tables"].keys()),
            "total_records": sum([info["row_count"] for info in schema["tables"].values()])
        })
    except Exception as e:
        print(f"Refresh error: {traceback.format_exc()}")
        return jsonify({"error": f"Schema refresh failed: {str(e)}"})

@app.route("/suggest", methods=["POST"])
def suggest_queries():
    """Suggest possible queries based on database content"""
    global assistant
    
    if not assistant:
        return jsonify({"error": "Please connect to database first"})
    
    try:
        # Get FRESH schema for suggestions
        schema = assistant.discover_database_schema(force_refresh=True)
        suggestions = []
        
        for table_name, table_info in schema["tables"].items():
            suggestions.extend([
                f"show all {table_name}",
                f"count {table_name}",
                f"show {table_name} columns",
            ])
            
            # Add column-specific suggestions
            for col in table_info["columns"]:
                if col["type"].lower().startswith(('varchar', 'text', 'char')):
                    suggestions.append(f"show unique {col['name']} from {table_name}")
                elif col["type"].lower().startswith(('int', 'decimal', 'float')):
                    suggestions.append(f"sum {col['name']} from {table_name}")
        
        return jsonify({"suggestions": suggestions[:10]})  # Limit to 10 suggestions
        
    except Exception as e:
        print(f"Suggestion error: {traceback.format_exc()}")
        return jsonify({"error": f"Suggestion generation failed: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True) 