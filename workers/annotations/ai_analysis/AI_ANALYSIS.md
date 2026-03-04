# AI Analysis Worker

This worker uses Claude (Anthropic's LLM) to perform natural language-driven analysis and manipulation of annotations, connections, and property values in a dataset. Users describe what they want in plain English, and the worker generates and executes Python code to carry it out.

## How It Works

1. **Data collection**: Fetches all annotations, connections, and property values from the dataset and assembles them into a dictionary structure and a pandas DataFrame.
2. **Prompt construction**: Builds a detailed prompt containing the user's query, available tags, property-to-tag mappings, DataFrame column names, and a sample of the data.
3. **Code generation**: Sends the prompt to Claude (claude-3-7-sonnet) with a system prompt that instructs it to write Python code operating on the dictionary and DataFrame.
4. **Execution**: Extracts the Python code block from Claude's response and executes it in a sandboxed namespace with access to numpy, pandas, shapely, scipy, and the annotation_tools utilities.
5. **Upload**: Deletes existing annotations and re-uploads the modified annotations, connections, and property values. Any new DataFrame columns are uploaded as a new AI-generated property.
6. **Documentation**: Saves timestamped JSON snapshots (before and after) and a text file with the full prompt and Claude's response to the dataset folder.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Query** | text | (none) | Natural language instruction describing the desired analysis or manipulation (e.g., "Color all cell objects red that are in the top 25 percentile by area"). |
| **AI Property Name** | text | `AI properties` | Name of the property group under which any new computed values will be stored. |
| **Output JSON filename** | text | `output.json` | Filename for the output JSON snapshot saved to the dataset folder. |
| **Claude API key** | text | (none) | Anthropic API key. Only shown if the `ANTHROPIC_API_KEY` environment variable is not set. |

## Implementation Details

- The worker operates on a `dictionary_data` structure containing `annotations`, `annotationConnections`, and `annotationPropertyValues`, plus a pandas DataFrame (`df`) with flattened property values indexed by annotation ID.
- The system prompt (loaded from `/system_prompt.txt` in the Docker image) teaches Claude the NimbusImage data schema, coordinate conventions (x/y swap), available helper functions from `annotation_tools`, and how to manipulate annotations, connections, colors, and properties.
- Generated code has access to numpy, pandas, geopandas, shapely, scipy (cKDTree, stats, optimize, interpolate), and annotation_tools.
- The worker creates an "AI property" in the dataset configuration if one does not already exist, and adds it to all dataset views so computed values are immediately visible.
- The update process deletes all existing annotations and re-creates them, remapping internal IDs for connections and property values to match the new server-assigned IDs.

## Notes

- Because the worker deletes and re-creates all annotations, it is destructive -- any annotations not present in the dictionary at execution time will be lost. Input/output JSON snapshots are saved for recovery.
- The LLM model used is `claude-3-7-sonnet-20250219` with `temperature=0` and `max_tokens=1103`.
- This worker is categorized as an annotation worker but can also create and modify property values and connections, making it a general-purpose analysis tool.
