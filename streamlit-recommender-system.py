import re
import streamlit as st
import pandas as pd
import numpy as np
import os
from terminal_langchain_user_prefs_feedback_learn_GPT_QUERY import TerminalRAG
from plotly_heatmap import create_plotly_interpolated_maps
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_key():
    # get key from Streamlit secrets (for deployment)
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    # fall back to environment variable (for local development)
    elif "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    else:
        st.error("No OpenAI API key found! Please add it to your environment or secrets.")
        return None

openai_api_key = get_api_key()

st.set_page_config(
    page_title="Workspace Recommender",
    page_icon="🏢",
    layout="wide"
)

def generate_visualizations(recommended_room=None):
    try:
        logger.info(f"Generating visualizations with recommended_room: {recommended_room}")
        if recommended_room:

            # Log room existence check
            if isinstance(recommended_room, list):
                for room in recommended_room:
                    if room in st.session_state.rag_system.sensor_data['Location'].values:
                        logger.info(f"Room {room} found in sensor data")
                    else:
                        logger.warning(f"Room {room} NOT FOUND in sensor data")
            else:
                if recommended_room in st.session_state.rag_system.sensor_data['Location'].values:
                    logger.info(f"Room {recommended_room} found in sensor data")
                else:
                    logger.warning(f"Room {recommended_room} NOT FOUND in sensor data")
        
        maps = create_plotly_interpolated_maps(
            sensor_df=st.session_state.rag_system.sensor_data,
            coord_df=st.session_state.rag_system.coord_data,
            parameters=["temperature_mean", "humidity_mean", "co2_mean", "light_mean", "pir_mean"],
            recommended_room=recommended_room,
            padding_percent=0.05
        )
        logger.info(f"Successfully generated {len(maps)} maps")
        return maps
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)
        st.error(f"Error generating visualizations: {str(e)}")
        return {}

# initialize session state for persistence
if 'rag_system' not in st.session_state:
    # first-time initialization
    st.session_state.rag_system = TerminalRAG()

    # set the API key
    from langchain_openai import ChatOpenAI
    st.session_state.rag_system.llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        openai_api_key=openai_api_key
    )

    st.session_state.rag_system.initialize_param_stats()
    st.session_state.rag_system.set_midpoint_defaults_for_generic_user()
    st.session_state.rag_system.initialize_chain()
    st.session_state.rag_system.authenticate_user()
    st.session_state.history = []
    st.session_state.has_recommendation = False
    
    # Generate initial maps
    st.session_state.maps = generate_visualizations()

# App header
st.title("Workspace Recommender")
st.markdown("""
This application is a recommender system based on the 2019 Oulu TellUs space sensor readings.  
It uses LLM chains to process user queries, generate database requests, analyze results, and provide relevant responses.  
The system can also dynamically adjust user preferences based on recent queries. Preferences are reset to default values for each new user.
""")

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    # Query input section
    st.subheader("What are you looking for?")

    # Store the current query in session state to persist it
    if 'query_text' not in st.session_state:
        st.session_state.query_text = ""
    
    # query = st.text_input(
    #     "Describe your ideal workspace:",
    #     placeholder="e.g., a quiet room with good lighting"
    # )
    
    example_queries = [
        "Find me a room with low CO2 levels",
        "I need a warm room with good lighting",
        "What's the quietest room available?",
        "I prefer a cooler environment with moderate humidity",
        "Show me the average temperature across all rooms",
        "I'd like a room that's slightly warmer than average"
    ]

    # When an example is selected, update the text input
    def set_query_from_example():
        if st.session_state.example_selector != "":
            st.session_state.query_text = st.session_state.example_selector

    # Example selector
    st.selectbox(
        "Try one of these examples:",
        [""] + example_queries,
        key="example_selector",
        on_change=set_query_from_example
    )

    # text input that uses the value from session state
    query = st.text_area(
        "Or type your own query:",
        value=st.session_state.query_text,
        height=100,  # Make it taller
        key="current_query",  # This connects it to session_state.query_text
        placeholder="e.g., a quiet room with good lighting"
    )
    
    
    #selected_example = st.selectbox(
    #    "Or try one of these examples:",
    #    [""] + example_queries
    #)

    # Update session state when text area changes
    if query != st.session_state.query_text:
        st.session_state.query_text = query
    
    submit_button = st.button("Find Workspace", type="primary")


    if submit_button:
        if query:
            with st.spinner("Analyzing your request..."):
                # Process the query
                answer = st.session_state.rag_system.ask_question(query)
                st.session_state.history.append({"query": query, "answer": answer})
                
                # Update maps based on recommendation
                recommended_rooms = []
                #recommended_room = None

                if st.session_state.rag_system.current_room_id:
                    recommended_rooms = [st.session_state.rag_system.current_room_id]
                    logger.info(f"Found room from current_room_id: {recommended_rooms}")
                    #recommended_room = st.session_state.rag_system.current_room_id

                if not recommended_rooms:
                # Try to parse from the answer text directly
                # Look for the "Selected_rooms:" pattern in the answer
                    import re
                    match = re.search(r"Selected_rooms:\s*(.*?)$", answer, re.MULTILINE)
                    if match:
                        # Split by commas and clean up
                        room_list = match.group(1).strip()
                        # Split and remove any trailing punctuation
                        recommended_rooms = []
                        for r in room_list.split(','):
                            # Strip whitespace and trailing punctuation
                            clean_room = re.sub(r'[^\w-]+$', '', r.strip())
                            if clean_room:
                                recommended_rooms.append(clean_room)

                # Try looking in last_recommended
                if not recommended_rooms and st.session_state.rag_system.last_recommended:
                    room_id = st.session_state.rag_system.last_recommended.get('Location')
                    if room_id:
                        recommended_rooms = [room_id]
                        logger.info(f"Found room from last_recommended: {recommended_rooms}")
            

                # validating rooms
                valid_rooms = []
                for room in recommended_rooms:
                    # Remove any trailing punctuation
                    clean_room = re.sub(r'[^\w-]+$', '', room)
                    if clean_room in st.session_state.rag_system.sensor_data['Location'].values:
                        valid_rooms.append(clean_room)
                    else:
                        logger.warning(f"Removing invalid room ID: {clean_room}")
                recommended_rooms = valid_rooms

                # limiting number to 5
                #MAX_ROOMS_TO_HIGHLIGHT = 5
                #if len(recommended_rooms) > MAX_ROOMS_TO_HIGHLIGHT:
                #    logger.info(f"Limiting visualization to {MAX_ROOMS_TO_HIGHLIGHT} out of {len(recommended_rooms)} rooms")
                #    recommended_rooms = recommended_rooms[:MAX_ROOMS_TO_HIGHLIGHT]

                logger.info(f"Query processed, recommended rooms: {recommended_rooms}")

                #elif st.session_state.rag_system.last_recommended:
                    # If there's a last_recommended but no current_room_id, use the location from last_recommended
                #    recommended_room = st.session_state.rag_system.last_recommended.get('Location')


                #recommended_room = st.session_state.rag_system.current_room_id
                #logger.info(f"Query processed, recommended room: {recommended_room}")
                
                if recommended_rooms:
                    st.session_state.has_recommendation = True
                    st.session_state.recommended_rooms = recommended_rooms
                    
                    # Generate new maps with recommendation highlighted
                    with st.spinner("Updating visualizations..."):
                        # Generate new maps with recommendation highlighted
                        st.session_state.maps = generate_visualizations(recommended_room=recommended_rooms)
                else:
                    st.session_state.has_recommendation = False
                    # Even without a recommendation, display maps without highlighting
                    st.session_state.maps = generate_visualizations()
        else:
            st.error("Please enter a query or select an example.")
    
    # Preference adjustment explanation
    with st.expander("How to adjust your preferences"):
        st.markdown("""
        You can adjust your preferences using natural language commands like:
        
        - "I prefer cooler rooms" or "Make it slightly warmer"
        - "I'm very sensitive to CO2" or "I don't mind humidity"
        - "Show me my current preferences"
        
        The system will automatically adjust your profile based on these requests.
        """)
    
    # Query history
    if st.session_state.get('history', []):
        st.subheader("Previous Queries")
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Q: {item['query']}"):
                st.markdown(f"**Answer:** {item['answer']}")

with col2:
    # Results area
    if st.session_state.get('history', []):
        latest = st.session_state.history[-1]
        
        # Show the answer
        st.markdown("### Recommendation")
        st.markdown(latest['answer'])
        
        # Debug info - temporarily show this to verify the recommendation is being passed
        # if 'recommended_room' in st.session_state:
        #    st.info(f"Current recommended room: {st.session_state.recommended_room}")
        
        # Show visualizations (either with or without recommendation)
        st.markdown("### Environment Visualization")
        
        # Display tabs with current visualizations
        tab_names = ["Temperature", "Humidity", "CO2", "Light", "Occupancy"]
        param_keys = ["temperature_mean", "humidity_mean", "co2_mean", "light_mean", "pir_mean"]
        
        tabs = st.tabs(tab_names)
        
        # Use the visualizations in each tab
        for i, tab in enumerate(tabs):
            with tab:
                param = param_keys[i]
                if param in st.session_state.maps:
                    st.plotly_chart(st.session_state.maps[param], use_container_width=True)
                else:
                    st.info(f"No visualization available for {tab_names[i]}")
        
        # Show room details if there's a recommendation
        if st.session_state.has_recommendation and st.session_state.rag_system.last_recommended:
            st.markdown("### Room Details")
            room_data = st.session_state.rag_system.last_recommended
            
            # Create columns for room metrics
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.metric("Temperature", f"{room_data['temperature_mean']:.1f}°C")
            with mc2:
                st.metric("Humidity", f"{room_data['humidity_mean']:.1f}%")
            with mc3:
                st.metric("CO2", f"{room_data['co2_mean']:.0f} ppm")
            with mc4:
                st.metric("Light", f"{room_data['light_mean']:.0f} lux")
            with mc5:
                st.metric("Occupancy", f"{room_data['pir_mean']:.1f}")
    else:
        # Initial state - show instructions and initial maps
        st.markdown("""
        ### Welcome to Workspace Recommender!
        
        Use the form on the left to describe your ideal workspace. You can:
        
        - Ask for rooms with specific environmental conditions
        - Request information about available spaces
        - Adjust your preferences using natural language
        - View visual representations of the recommended spaces
        """)
        
        # Show initial maps
        st.markdown("### Current Environment Conditions")
        
        # Display tabs with initial visualizations
        tab_names = ["Temperature", "Humidity", "CO2", "Light", "Occupancy"]
        param_keys = ["temperature_mean", "humidity_mean", "co2_mean", "light_mean", "pir_mean"]
        
        tabs = st.tabs(tab_names)
        
        # Use the initial maps in each tab
        for i, tab in enumerate(tabs):
            with tab:
                param = param_keys[i]
                if 'maps' in st.session_state and param in st.session_state.maps:
                    st.plotly_chart(st.session_state.maps[param], use_container_width=True)
                else:
                    st.info(f"Initial visualization not available for {tab_names[i]}")

# Footer
st.markdown("---")
st.markdown("Workspace Recommender | Powered by AI")