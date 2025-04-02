import os
import pandas as pd
import re
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from preference_integration import PreferenceAdjuster
from feedback_tracker import FeedbackTracker
from heat_map import plot_rectangular_heatmaps_for_parameters, plot_all_parameters_by_coordinates, plot_multiple_interpolations
import time
import traceback
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)

load_dotenv()

HARDCODED_CORNERS = [(65.05892835, 25.46471000), (65.05860430, 25.46470780), (65.05860348, 25.46612754), (65.05892835, 25.46612620)]



class TerminalRAG:
    def __init__(self):
        self.user_id = None
        self.user_preferences = pd.read_csv("user_preferences.csv")
        self.sensor_data = pd.read_csv("summary_stats.csv")
        self.coord_data = pd.read_excel("Coords.xlsx")
        #self.feedback_tracker = FeedbackTracker()
        self.current_room_id = None
        #self.preference_adjuster = PreferenceAdjuster()
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        self.query_chain = None
        self.analysis_chain = None
        self.recommended_room = None
        self.last_recommended = None
        self.param_stats = {}

    def initialize_param_stats(self):
        """
        Compute min, max, and stdev for each parameter in sensor_data.
        pass these stats to the LLM so it can decide partial offsets or skipping.
        """
        # For each param: temperature_mean, humidity_mean, co2_mean, light_mean, pir_mean
        # store them in a dictionary: self.param_stats["temperature"] = {...}
        self.param_stats["temperature"] = {
            "min": float(self.sensor_data["temperature_mean"].min()),
            "max": float(self.sensor_data["temperature_mean"].max()),
            "std": float(self.sensor_data["temperature_mean"].std())
        }
        self.param_stats["humidity"] = {
            "min": float(self.sensor_data["humidity_mean"].min()),
            "max": float(self.sensor_data["humidity_mean"].max()),
            "std": float(self.sensor_data["humidity_mean"].std())
        }
        self.param_stats["co2"] = {
            "min": float(self.sensor_data["co2_mean"].min()),
            "max": float(self.sensor_data["co2_mean"].max()),
            "std": float(self.sensor_data["co2_mean"].std())
        }
        self.param_stats["light"] = {
            "min": float(self.sensor_data["light_mean"].min()),
            "max": float(self.sensor_data["light_mean"].max()),
            "std": float(self.sensor_data["light_mean"].std())
        }
        self.param_stats["occupancy"] = {
            "min": float(self.sensor_data["pir_mean"].min()),
            "max": float(self.sensor_data["pir_mean"].max()),
            "std": float(self.sensor_data["pir_mean"].std())
        }

    def set_midpoint_defaults_for_generic_user(self):
        """
        Overwrites/creates the generic_user row so that each preference is at the midpoint of [min,max].
        So we never start out-of-bounds.
        """

        temperature_mid = (self.param_stats["temperature"]["min"] + self.param_stats["temperature"]["max"]) / 2
        humidity_mid = (self.param_stats["humidity"]["min"] + self.param_stats["humidity"]["max"]) / 2
        co2_mid = (self.param_stats["co2"]["min"] + self.param_stats["co2"]["max"]) / 2
        light_mid = (self.param_stats["light"]["min"] + self.param_stats["light"]["max"]) / 2
        occupancy_mid = (self.param_stats["occupancy"]["min"] + self.param_stats["occupancy"]["max"]) / 2

        default_dict = {
            "user_id": "generic_user",
            "temperature_preference": temperature_mid,
            "temperature_sensitivity": "normal",
            "temperature_min": self.param_stats["temperature"]["min"],
            "temperature_max": self.param_stats["temperature"]["max"],

            "humidity_preference": humidity_mid,
            "humidity_sensitivity": "normal",
            "humidity_min": self.param_stats["humidity"]["min"],
            "humidity_max": self.param_stats["humidity"]["max"],

            "co2_preference": co2_mid,
            "co2_sensitivity": "normal",
            "co2_min": self.param_stats["co2"]["min"],
            "co2_max": self.param_stats["co2"]["max"],

            "light_preference": light_mid,
            "light_sensitivity": "normal",
            "light_min": self.param_stats["light"]["min"],
            "light_max": self.param_stats["light"]["max"],

            "occupancy_preference": occupancy_mid,
            "occupancy_sensitivity": "normal",
            "occupancy_min": self.param_stats["occupancy"]["min"],
            "occupancy_max": self.param_stats["occupancy"]["max"]
        }

        # overwrite or create that row in user_preferences
        if "generic_user" not in self.user_preferences["user_id"].values:
            self.user_preferences = pd.concat([
                self.user_preferences,
                pd.DataFrame([default_dict])
            ], ignore_index=True)
        else:
            idx = self.user_preferences["user_id"]=="generic_user"
            for k,v in default_dict.items():
                self.user_preferences.loc[idx, k] = v

        self.user_preferences.to_csv("user_preferences.csv", index=False)
        print("Set 'generic_user' defaults to dataset midpoints for each param!")

    def initialize_chain(self):
        print("Creating LLM chains...")
        
        # Chain for generating pandas queries
        query_generation_prompt = PromptTemplate(
            template = """
            You are a data query assistant that converts natural language questions about location sensor data 
            into a single-line Pandas DataFrame operation on 'df'. 

            - The DataFrame 'df' has columns:
            Location (str),
            temperature_mean (float),
            humidity_mean (float),
            co2_mean (float),
            light_mean (float),
            pir_mean (float),
            description (str)

            We have these dataset stats (min, max, std) in JSON: {param_stats_json}

            When users mention words like:
            - "colder" or "cooler" => interpret as near the lower end (above min but below midpoint or stdev) of the temperature range
            - "warmer" or "hotter" => near the higher end (below max but above midpoint) of the temperature range
            - "dim" => near the lower end of light_mean
            - "bright" => near the upper end of light_mean
            - "quiet" => near the lower end of pir_mean (occupancy)
            - "crowded" => near the higher end of pir_mean
            - "good" or "bad" => interpret in context of param_stats if user specifically refers to "good temperature" => near comfortable mid range, "bad humidity" => near extremes, etc.
            These are just examples, apply this logic to all words users might describe when talking about location parameters. 

            **Always respect the min and max from param_stats_json**. 
            Example: If user says "hotter than 30" but max is 25.2, fallback to 25.2 and notify user that their request cannot be fulfilled but say that this is the closest match available. 


            ALSO: If the user wants a general statistic such as average or count - e.g. 
            "What is the average temperature across all locations?" - produce an 
            aggregation snippet (like df["temperature_mean"].mean() or df.shape[0], etc.)

            - The user has existing preferences: {user_preferences}
            - Last recommended location (or no location): {last_recommendation_context}

            - The user's question might ask for "colder location", "lowest CO2", or 
            "average temperature," or "how many locations have humidity>60," etc.

            - Return ONLY a single-line Pandas code snippet that, when evaluated, 
            produces the correct result. No text or explanation, just code.

            - If there is a last recommended location, you must use that location's metrics as 
            the primary reference instead of the user preferences, if the user specifically 
            references "colder/warmer than last recommended," etc.

            - If your strict filter yields zero rows, switch to a fallback distance approach 
            and pick the top 5. Example:
                df.assign(distance = abs(df['temperature_mean'] - 22)+...).sort_values('distance').head(5)

            - Do not reference any variables except 'df' or literal numbers. 
            Do not use variables like 'preferences[...]', etc.

            User's question: "{question}"
            """,
            input_variables=["question", "user_preferences", "last_recommendation_context", "param_stats_json"]
        )
        self.query_chain = LLMChain(llm=self.llm, prompt=query_generation_prompt)
        
        # Chain for analyzing results
        analysis_prompt = PromptTemplate(
            template="""
            You are an AI assistant specialized in analyzing sensor data for locations and providing personalized recommendations.
            Provide answers in a single concise sentence whenever possible.

            SYSTEM CONTEXT:
            - We have top_locations_json: {top_rooms_json}
            - The user's preference info is: {context}
            - The user's question is: {question}
            - Current user ID: {current_user}

            Instructions:
            1. If top_locations_json only has aggregator data and top_locations_json is an array with a single dictionary that has a "Value" key, response should include <that numeric> as it is likely the answer. 
            2. If it shows multiple locations but user only asked for one location, pick exactly one as recommended (or none if no suitable).
            3. If user specifically asks for multiple locations (like "several" or "top 3"), you can produce as many as the user needs. 
            If the user asks for too many, apologize for the inconvenience and give the maximum number that you have. State that you cannot display this many locations.

            CRITICAL: For ANY location or locations you identify in your answer, you MUST include exactly one line at the end in this format: Selected_locations: <location1_id>, <location2_id>, <location3_id>
            in a separate line at the end. 
                - No periods after location IDs
                - Separate multiple locations with commas only
                - List at most 5 specific locations, even if more are found
                - If more than 5 locations match, mention something like this in your response: "Showing 5 out of X matching locations"
            
            4. If no locations are returned, handle gracefully.

            Don't disclaim you lack user preference data — you have it in {context}.
            """,
            input_variables=["context", "question", "current_user", "top_rooms_json"]
        )
        self.analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)

        preference_parser_prompt = PromptTemplate(
            template="""
                You are a "Preference Parser" for a sensor-based recommendation system.

                **IMPORTANT**: Return valid JSON *only*, with *no* triple backticks, code fences, or any extra text. 
                Only a single JSON object as your entire output.

                We have the dataset stats (min, max, stdev) for each param in JSON: {param_stats_json}

                The user's current preferences in JSON: {current_prefs_json}

                The user's latest message is: "{user_text}"

                Instructions:
                1. If the user says "slightly warmer," interpret that as +0.5 * stdev for temperature, 
                if "much warmer," consider +2 or +3 stdev if feasible, etc. 
                2. If the offset or absolute value goes beyond min-max, either do partial offset or do "action":"none" 
                if it's still not feasible. 
                3. If user only wants to show preferences, set "action":"show," everything else null. 
                4. If user says "colder than last recommended reading by X," do the math. 
                E.g. last reading=21.3 => new temperature_preference=21.3 - X if in range. 
                5. If user does not mention any preference change, set "action":"none." 
                6. Return EXACT JSON with the shape:

                {{
                "action": "<'update','show','none'>",
                "temperature_preference": <float or null>,
                "temperature_sensitivity": <string or null>,
                "temperature_min": <float or null>,
                "temperature_max": <float or null>,

                "humidity_preference": <float or null>,
                "humidity_sensitivity": <string or null>,
                "humidity_min": <float or null>,
                "humidity_max": <float or null>,

                "co2_preference": <float or null>,
                "co2_sensitivity": <string or null>,
                "co2_min": <float or null>,
                "co2_max": <float or null>,

                "light_preference": <float or null>,
                "light_sensitivity": <string or null>,
                "light_min": <float or null>,
                "light_max": <float or null>,

                "occupancy_preference": <float or null>,
                "occupancy_sensitivity": <string or null>,
                "occupancy_min": <float or null>,
                "occupancy_max": <float or null>,

                "notes": "<brief explanation of how you computed the final numeric>"
                }}

                No extra lines, no triple backticks. Only valid JSON.        
                """,
                input_variables=["user_text","param_stats_json", "current_prefs_json"]
        )
        self.preference_parser_chain = LLMChain(
            llm=self.llm, 
            prompt=preference_parser_prompt
        )
        print("Chains created!")


    def clean_code_snippet(self, raw_code: str) -> str:
        code = raw_code.strip()

        if code.startswith("df["):
            inside = code[2:]  # skip the 'df'
            # if it starts with '[' but doesn't contain another '[' => aggregator
            # if it starts with '[' and DOES contain another '[' => filter

            if '[' in inside[1:]:  # check after the first bracket
                # e.g. df[df['temp']>20]
                code = "df.loc" + code[2:]
            else:
                # aggregator code => do nothing
                pass

        elif code.startswith("df."):
            pass
        elif code.startswith(".") or code.startswith("["):
            code = "df" + code

        return code
    
    def parse_preferences_intent_and_changes(self, user_message: str):
        """
        1) Build stats_json from self.param_stats
        2) Run preference_parser_chain
        3) Return (action, changes)
        """
        # build param_stats_json
        param_stats_json = json.dumps(self.param_stats, default=str)

        # build current_prefs for this user
        current_prefs = self.user_preferences[self.user_preferences["user_id"]==self.user_id].iloc[0].to_dict()
        current_prefs_json = json.dumps(current_prefs, default=str)

        parse_result = self.preference_parser_chain.run({
            "user_text": user_message,
            "param_stats_json": param_stats_json,
            "current_prefs_json": current_prefs_json
            
        })
        print("Debug: preference parser raw output =>", parse_result)

        try:
            data = json.loads(parse_result)
        except json.JSONDecodeError:
            print("Could not parse preference JSON. Interpreting as no preference action.")
            return ("none", {})

        action = data.get("action","none")
        columns = [
            "temperature_preference","temperature_sensitivity","temperature_min","temperature_max",
            "humidity_preference","humidity_sensitivity","humidity_min","humidity_max",
            "co2_preference","co2_sensitivity","co2_min","co2_max",
            "light_preference","light_sensitivity","light_min","light_max",
            "occupancy_preference","occupancy_sensitivity","occupancy_min","occupancy_max"
        ]
        changes = {}
        if action=="update":
            for col in columns:
                val = data.get(col, None)
                if val not in [None,"null"]:
                    changes[col] = val

        return (action, changes)
    

    def reset_generic_user(self):
        """
        Force-reset the 'generic_user' row in user_preferences to default values
        every time we start up, so changes from previous runs are wiped out.
        """
        # If 'generic_user' doesn't exist, create it. Otherwise, overwrite it.
        if "generic_user" not in self.user_preferences["user_id"].values:
            # Insert a new row
            new_row = {"user_id": "generic_user"}
            new_row.update(GENERIC_DEFAULTS)
            self.user_preferences = pd.concat([
                self.user_preferences,
                pd.DataFrame([new_row])
            ], ignore_index=True)
        else:
            # Overwrite existing columns
            idx = self.user_preferences["user_id"] == "generic_user"
            for col, val in GENERIC_DEFAULTS.items():
                self.user_preferences.loc[idx, col] = val

        self.user_preferences.to_csv("user_preferences.csv", index=False)
        print("Reset generic_user to default values!")

    def parse_recommended_room(self, llm_answer: str):
        """
        Looks for a line in the LLM's final text like 'RECOMMENDED_ROOM: SomeRoomID'.
        Returns the room ID or None if not found.
        """
        match = re.search(r"Selected_locations:\s*([^\s]+)", llm_answer)
        if match:
            return match.group(1).strip()
        return None
    
    def authenticate_user(self):
        self.user_id = "generic_user"
        print(f"Using standard user profile for testing!")
        self.show_user_profile()
    
    """
    def authenticate_user(self):
        while True:
            user_id = input("Enter your user ID (e.g., user_1): ")
            if user_id in self.user_preferences['user_id'].values:
                self.user_id = user_id
                print(f"Welcome, {user_id}!")
                self.show_user_profile()
                break
            print("User not recognized. Please try again.")
    """

    def show_user_profile(self):
        return


    def is_recommendation(self, question, answer):
        recommendation_keywords = ['recommend', 'best', 'suitable', 'good for', 'suggest']
        return any(keyword in question.lower() for keyword in recommendation_keywords) or 'room' in answer.lower()
    

    def update_user_preferences(self, changes: dict):
        param_map = {
            "temperature_preference": "temperature",
            "humidity_preference": "humidity",
            "co2_preference": "co2",
            "light_preference": "light",
            "occupancy_preference": "occupancy"
        }

        idx = self.user_preferences["user_id"] == "generic_user"

        for col, val in changes.items():
            if col in param_map:
                param = param_map[col]
                p_min = self.param_stats[param]["min"]
                p_max = self.param_stats[param]["max"]
                fval = float(val)
                if fval < p_min or fval > p_max:
                    print(f"User requested {col}={fval}, out of dataset range {p_min}-{p_max}. Skipping this param.")
                    continue
                self.user_preferences.loc[idx, col] = fval
                print(f"Updated {col} => {fval}")
            else:
                # if it's e.g. temperature_sensitivity, or min / max, or a non-‘preference’ column
                self.user_preferences.loc[idx, col] = val
                print(f"Updated {col} => {val}")

    def display_user_preferences(self):
        # get user row
        row = self.user_preferences[self.user_preferences["user_id"] == self.user_id].iloc[0]
        print("\nYour Current Preferences:")
        print(f"Temperature: {row['temperature_preference']} (sensitivity: {row['temperature_sensitivity']})")
        print(f"   Range: min={row['temperature_min']}, max={row['temperature_max']}")
        print(f"Humidity: {row['humidity_preference']} (sensitivity: {row['humidity_sensitivity']})")
        print(f"   Range: min={row['humidity_min']}, max={row['humidity_max']}")
        print(f"CO2: {row['co2_preference']} (sensitivity: {row['co2_sensitivity']})")
        print(f"   Range: min={row['co2_min']}, max={row['co2_max']}")
        print(f"Light: {row['light_preference']} (sensitivity: {row['light_sensitivity']})")
        print(f"   Range: min={row['light_min']}, max={row['light_max']}")
        print(f"Occupancy: {row['occupancy_preference']} (sensitivity: {row['occupancy_sensitivity']})")
        print(f"   Range: min={row['occupancy_min']}, max={row['occupancy_max']}")
        print()

    
    def ask_question(self, question: str):
        try:
            print("\nDebug: Starting ask_question")
            print(f"Debug: Question received: {question}")

            # 1) interpret preference changes
            action, changes = self.parse_preferences_intent_and_changes(question)
            if action=="show":
                self.display_user_preferences()
            elif action=="update" and len(changes)>0:
                self.update_user_preferences(changes)

            # 2) aggregator or recommended-room
            if self.user_id:
                user_data = self.user_preferences[self.user_preferences["user_id"]==self.user_id].iloc[0]
                user_profile = (
                    f"=== CURRENT USER: {self.user_id} ===\n"
                    f"- Temperature: {user_data['temperature_preference']}\n"
                    f"- Humidity: {user_data['humidity_preference']}\n"
                    f"- CO2: {user_data['co2_preference']}\n"
                    f"- Light: {user_data['light_preference']}\n"
                    f"- Occupancy: {user_data['occupancy_preference']}\n"
                )
            else:
                user_profile = "NO USER LOGGED IN\n"
                user_data = None

            # see if we want to do a data snippet
            need_data = any(k in question.lower() for k in ["room","temperature","humidity","co2","light","occupancy"])
            top_rooms_json = "[]"
            relevant_rooms = pd.DataFrame()

            # build last recommendation context
            if self.last_recommended:
                context_string = f"Previously recommended = {self.last_recommended['Location']}"
            else:
                context_string = "No previously recommended room."

            # for passing dataset stats into the query chain
            stats_str = json.dumps(self.param_stats, default=str)

            if need_data and user_data is not None:
                query_result = self.query_chain.invoke({
                    "question": question,
                    "user_preferences": user_data.to_dict(),
                    "last_recommendation_context": context_string,
                    "param_stats_json": stats_str
                })
                raw_code = query_result["text"].strip()
                print("Debug: LLM raw code:", raw_code)

                final_code = raw_code  # we skip any .loc insertion
                print("Debug: final code snippet:", final_code)

                try:
                    df = self.sensor_data
                    result = eval(final_code)
                    if isinstance(result, pd.DataFrame):
                        if result.empty:
                            print("No rooms found for that filter. Possibly the user requested an offset that yields zero rows.")
                            top_rooms_json = "[]"
                        else:
                            columns_to_show = ["Location","temperature_mean","humidity_mean","co2_mean"]
                            if "score" in result.columns:
                                columns_to_show.append("score")
                            print("Debug: relevant_rooms data:")
                            # To avoid exceeding context length
                            truncated = result.head(10)
                            print(result[columns_to_show])
                            top_rooms_json = json.dumps(truncated.to_dict(orient="records"))

                    elif isinstance(result,(int,float)):
                        print(f"Debug: aggregator result => {result}")
                        top_rooms_json = json.dumps([{"Value": result}])
                    elif isinstance(result,pd.Series):
                        as_df = result.to_frame("Value").reset_index()
                        print(as_df)
                        top_rooms_json = as_df.to_json(orient="records")
                    else:
                        print("Unknown snippet result type, returning empty JSON")
                        top_rooms_json = "[]"

                except Exception as e:
                    print(f"Debug: Query execution failed => {e}")
                    top_rooms_json="[]"

            # now do analysis chain
            context_for_analysis = user_profile
            analysis_result = self.analysis_chain.invoke({
                "question": question,
                "context": context_for_analysis,
                "current_user": self.user_id,
                "top_rooms_json": top_rooms_json
            })
            final_answer = analysis_result["text"]
            print("Debug: analysis chain output =>", final_answer)

            recommended_id = self.parse_recommended_room(final_answer)
            if recommended_id:
                # store last recommended
                row_df = self.sensor_data[self.sensor_data["Location"]==recommended_id]
                if row_df.empty:
                    self.last_recommended=None
                    self.current_room_id=None
                else:
                    self.last_recommended = row_df.iloc[0].to_dict()
                    self.current_room_id = recommended_id
                    print(f"Debug: LLM recommended room => {recommended_id}")
            else:
                self.last_recommended=None
                self.current_room_id=None

            return final_answer

        except Exception as e:
            print(f"Debug: Full error => {str(e)}")
            print(f"Traceback =>\n{traceback.format_exc()}")
            return None


    def process_feedback(self, feedback_text, room_conditions):
        is_positive = self.llm.invoke(
            f"Analyze if this feedback is positive or negative: '{feedback_text}'. "
            "Respond with only 'positive' or 'negative'."
        ).content.strip().lower() == 'positive'

        if is_positive:
            self.feedback_tracker.add_feedback(self.user_id, self.current_room_id, 1)
            print("Thank you for your positive feedback!")
            return

        adjusted, changes = self.preference_adjuster.adjust_preference(
            self.user_id,
            room_conditions,
            is_positive
        )

        if adjusted:
            print("\nAdjustments made to your preferences:")
            for category, change in changes.items():
                print(f"{category}: {change['from']} → {change['to']}")
            self.feedback_tracker.add_feedback(self.user_id, self.current_room_id, 0)
        else:
            print("No preference adjustments needed based on your feedback.")

    def get_room_conditions(self, room_id):
        room_data = self.sensor_data[self.sensor_data['Location'] == room_id].iloc[0]
        return self.preference_adjuster.analyze_room_conditions(self.sensor_data, room_id)

    def batch_init(self, user_id=None):
        """Initialize for batch processing with specific user"""
        self.user_id = "generic_user"  # Always use generic_user regardless of input
        if not hasattr(self, 'query_chain') or self.query_chain is None:
            self.initialize_chain()
        return True
    
    #def batch_init(self, user_id):
    #    """Initialize for batch processing with specific user"""
    #    self.user_id = user_id
    #    if not hasattr(self, 'query_chain') or self.query_chain is None:
    #        self.initialize_chain()
    #    return True

    def run(self):
        print("Initializing system...")
        # init param stats
        self.initialize_param_stats()
        self.set_midpoint_defaults_for_generic_user()

        #self.reset_generic_user()
        # show maps
        plot_rectangular_heatmaps_for_parameters(df=self.sensor_data, recommended_room=None)
        plot_all_parameters_by_coordinates(
            sensor_df=self.sensor_data,
            coord_df=self.coord_data,
            parameters=["temperature_mean","humidity_mean","co2_mean","light_mean","pir_mean"],
            recommended_room=None,
            corners=HARDCODED_CORNERS
        )
        plot_multiple_interpolations(
            sensor_df=self.sensor_data,
            coord_df=self.coord_data,
            parameters=["temperature_mean","humidity_mean","co2_mean","light_mean","pir_mean"],
            recommended_room=None,
            padding_percent=0.05
        )

        self.initialize_chain()
        self.authenticate_user()

        while True:
            print("\nEnter your question (or 'quit' to exit):")
            user_q = input("> ")
            if user_q.lower()=="quit":
                break

            answer = self.ask_question(user_q)
            if answer:
                print("\nAnswer:", answer)
                if self.current_room_id:
                    plot_rectangular_heatmaps_for_parameters(
                        df=self.sensor_data,
                        recommended_room=self.current_room_id
                    )
                    plot_multiple_interpolations(
                        sensor_df=self.sensor_data,
                        coord_df=self.coord_data,
                        parameters=["temperature_mean","humidity_mean","co2_mean","light_mean","pir_mean"],
                        recommended_room=self.current_room_id,
                        padding_percent=0.05
                    )
                else:
                    print("No recommended room identified.")

            print("\nExample questions:")
            print("1. What room would you recommend for someone sensitive to humidity?")
            print("2. Which room has the best conditions for me?")
            print("3. Can you suggest a quiet room with good lighting?")
            print("4. What's the average temperature across all rooms?")


if __name__ == "__main__":
    rag_system = TerminalRAG()
    rag_system.run()