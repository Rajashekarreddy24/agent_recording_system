from django.test import TestCase, Client

class APITests(TestCase):
    def test_recording_api(self):
        client = Client()
        agent_name = "human"
        
        # Start recording
        start_data = {
            "ticket_id": "1",
            "fps": 30,
            "resolution": [1920, 1080],
            "metadata": {
                "browser": "Chrome",
                "os": "Windows"
            }
        }
        start_response = client.post(
            f"/recordings/agent/start-recording/{agent_name}/",
            data=start_data,
            content_type="application/json"  # Required for JSON data
        )
        print("Start recording response:", start_response.json())
        
        self.assertEqual(start_response.status_code, 200)
        recording_id = start_response.json().get("recording_id")
        
        if recording_id:
            # Check status
            status_response = client.get(
                f"/recordings/recording-status/{recording_id}/"
            )
            print("Status response:", status_response.json())
            self.assertEqual(status_response.status_code, 200)
            
            # Stop recording
            stop_response = client.post(
                f"/recordings/agent/stop-recording/{agent_name}/",
                content_type="application/json"
            )
            print("Stop recording response:", stop_response.json())
            self.assertEqual(stop_response.status_code, 200)

