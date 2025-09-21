"""Staff agent implementations for the AI Agent Dinosaur Simulator."""

import logging
from typing import Dict, List, Optional, Any
from agents.base_agent import DinosaurAgent
from models.core import Agent
from models.config import OpenAIConfig, AG2Config, Location
from models.enums import AgentRole, AgentState, PersonalityTrait, EventType


class ParkRangerAgent(DinosaurAgent):
    """Park ranger agent specialized in wildlife management and visitor safety."""
    
    def __init__(self, agent_model: Agent, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize park ranger agent.
        
        Args:
            agent_model: The agent data model
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        super().__init__(agent_model, openai_config, ag2_config)
        self.expertise_areas = ["wildlife_behavior", "safety_protocols", "emergency_response"]
        self.logger = logging.getLogger(f"{__name__}.ParkRanger.{agent_model.name}")
    
    def assess_dinosaur_threat(self, dinosaur_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess threat level of a dinosaur situation.
        
        Args:
            dinosaur_info: Information about the dinosaur and situation
            
        Returns:
            Threat assessment with recommended actions
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            # Create assessment prompt
            assessment_prompt = (
                f"As an experienced park ranger, assess the threat level of this dinosaur situation: "
                f"{dinosaur_info}. Provide a threat level (1-10), immediate actions needed, "
                f"and safety recommendations for visitors and staff."
            )
            
            response = self.generate_reply(
                messages=[{"content": assessment_prompt, "role": "user", "name": "System"}]
            )
            
            self.logger.info(f"Park ranger {self.agent_model.name} assessed dinosaur threat")
            
            return {
                "assessment": response,
                "ranger_id": self.agent_model.id,
                "expertise": "wildlife_management",
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error in threat assessment: {e}")
            return {
                "assessment": "Unable to complete threat assessment at this time",
                "error": str(e)
            }
        finally:
            self.update_state(AgentState.IDLE)
    
    def coordinate_evacuation(self, area: str, reason: str) -> str:
        """Coordinate visitor evacuation from an area.
        
        Args:
            area: Area to evacuate
            reason: Reason for evacuation
            
        Returns:
            Evacuation coordination response
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            evacuation_prompt = (
                f"As a park ranger, coordinate the evacuation of {area} due to {reason}. "
                f"Provide clear instructions for visitors and staff, establish safe zones, "
                f"and outline the evacuation procedure."
            )
            
            response = self.generate_reply(
                messages=[{"content": evacuation_prompt, "role": "user", "name": "EmergencySystem"}]
            )
            
            self.logger.info(f"Park ranger {self.agent_model.name} coordinated evacuation of {area}")
            return response
        
        except Exception as e:
            self.logger.error(f"Error coordinating evacuation: {e}")
            return "Unable to coordinate evacuation properly. Implementing emergency protocols."
        finally:
            self.update_state(AgentState.IDLE)
    
    def provide_safety_briefing(self, visitor_group: List[str]) -> str:
        """Provide safety briefing to a group of visitors.
        
        Args:
            visitor_group: List of visitor IDs
            
        Returns:
            Safety briefing content
        """
        briefing_prompt = (
            f"As a park ranger, provide a comprehensive safety briefing for {len(visitor_group)} visitors. "
            f"Cover dinosaur safety protocols, emergency procedures, and park rules. "
            f"Make it engaging but emphasize the importance of following safety guidelines."
        )
        
        try:
            response = self.generate_reply(
                messages=[{"content": briefing_prompt, "role": "user", "name": "VisitorServices"}]
            )
            
            self.logger.info(f"Park ranger {self.agent_model.name} provided safety briefing")
            return response
        
        except Exception as e:
            self.logger.error(f"Error providing safety briefing: {e}")
            return "Please follow all posted safety signs and stay with your group at all times."


class VeterinarianAgent(DinosaurAgent):
    """Veterinarian agent specialized in dinosaur health and medical care."""
    
    def __init__(self, agent_model: Agent, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize veterinarian agent.
        
        Args:
            agent_model: The agent data model
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        super().__init__(agent_model, openai_config, ag2_config)
        self.medical_specialties = ["dinosaur_physiology", "emergency_medicine", "behavioral_health"]
        self.logger = logging.getLogger(f"{__name__}.Veterinarian.{agent_model.name}")
    
    def diagnose_dinosaur_condition(self, symptoms: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose a dinosaur's medical condition based on symptoms.
        
        Args:
            symptoms: Dictionary of observed symptoms and behaviors
            
        Returns:
            Diagnosis and treatment recommendations
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            diagnosis_prompt = (
                f"As a veterinarian specializing in dinosaur medicine, analyze these symptoms: {symptoms}. "
                f"Provide a differential diagnosis, recommended treatments, and any immediate care needed. "
                f"Consider the unique physiology and needs of the dinosaur species involved."
            )
            
            response = self.generate_reply(
                messages=[{"content": diagnosis_prompt, "role": "user", "name": "MedicalSystem"}]
            )
            
            self.logger.info(f"Veterinarian {self.agent_model.name} provided diagnosis")
            
            return {
                "diagnosis": response,
                "veterinarian_id": self.agent_model.id,
                "specialty": "dinosaur_medicine",
                "urgency_level": self._assess_urgency(symptoms),
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error in diagnosis: {e}")
            return {
                "diagnosis": "Unable to complete diagnosis. Recommend immediate observation and basic care.",
                "error": str(e)
            }
        finally:
            self.update_state(AgentState.IDLE)
    
    def _assess_urgency(self, symptoms: Dict[str, Any]) -> str:
        """Assess the urgency level of symptoms.
        
        Args:
            symptoms: Dictionary of symptoms
            
        Returns:
            Urgency level (low, medium, high, critical)
        """
        # Simple urgency assessment based on keywords
        critical_keywords = ["bleeding", "unconscious", "seizure", "difficulty_breathing"]
        high_keywords = ["aggressive", "fever", "limping", "vomiting"]
        
        symptom_text = str(symptoms).lower()
        
        if any(keyword in symptom_text for keyword in critical_keywords):
            return "critical"
        elif any(keyword in symptom_text for keyword in high_keywords):
            return "high"
        elif len(symptoms) > 3:
            return "medium"
        else:
            return "low"
    
    def administer_treatment(self, treatment_plan: str, dinosaur_id: str) -> str:
        """Administer treatment to a dinosaur.
        
        Args:
            treatment_plan: Detailed treatment plan
            dinosaur_id: ID of the dinosaur being treated
            
        Returns:
            Treatment administration report
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            treatment_prompt = (
                f"As a veterinarian, implement this treatment plan for dinosaur {dinosaur_id}: {treatment_plan}. "
                f"Describe the treatment process, any complications encountered, and the dinosaur's response. "
                f"Include post-treatment care instructions and monitoring requirements."
            )
            
            response = self.generate_reply(
                messages=[{"content": treatment_prompt, "role": "user", "name": "TreatmentSystem"}]
            )
            
            self.logger.info(f"Veterinarian {self.agent_model.name} administered treatment to {dinosaur_id}")
            return response
        
        except Exception as e:
            self.logger.error(f"Error administering treatment: {e}")
            return "Treatment could not be completed as planned. Monitoring patient closely."
        finally:
            self.update_state(AgentState.IDLE)
    
    def conduct_health_checkup(self, dinosaur_id: str) -> Dict[str, Any]:
        """Conduct routine health checkup for a dinosaur.
        
        Args:
            dinosaur_id: ID of the dinosaur
            
        Returns:
            Health checkup report
        """
        checkup_prompt = (
            f"Conduct a routine health checkup for dinosaur {dinosaur_id}. "
            f"Assess vital signs, behavior, appetite, and overall condition. "
            f"Provide recommendations for ongoing care and note any concerns."
        )
        
        try:
            response = self.generate_reply(
                messages=[{"content": checkup_prompt, "role": "user", "name": "HealthSystem"}]
            )
            
            self.logger.info(f"Veterinarian {self.agent_model.name} completed checkup for {dinosaur_id}")
            
            return {
                "checkup_report": response,
                "dinosaur_id": dinosaur_id,
                "veterinarian_id": self.agent_model.id,
                "checkup_date": self.agent_model.last_activity,
                "next_checkup_recommended": "30 days"
            }
        
        except Exception as e:
            self.logger.error(f"Error conducting checkup: {e}")
            return {
                "checkup_report": "Checkup could not be completed. Schedule follow-up examination.",
                "error": str(e)
            }


class SecurityAgent(DinosaurAgent):
    """Security agent specialized in park security and visitor protection."""
    
    def __init__(self, agent_model: Agent, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize security agent.
        
        Args:
            agent_model: The agent data model
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        super().__init__(agent_model, openai_config, ag2_config)
        self.security_clearance = "high"
        self.response_protocols = ["crowd_control", "emergency_response", "threat_neutralization"]
        self.logger = logging.getLogger(f"{__name__}.Security.{agent_model.name}")
    
    def assess_security_threat(self, threat_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess a security threat and determine response level.
        
        Args:
            threat_info: Information about the potential threat
            
        Returns:
            Threat assessment and response recommendations
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            assessment_prompt = (
                f"As a security officer, assess this potential threat: {threat_info}. "
                f"Determine the threat level (1-10), immediate response needed, "
                f"and resources required. Consider visitor safety as the top priority."
            )
            
            response = self.generate_reply(
                messages=[{"content": assessment_prompt, "role": "user", "name": "SecuritySystem"}]
            )
            
            self.logger.info(f"Security agent {self.agent_model.name} assessed threat")
            
            return {
                "threat_assessment": response,
                "security_officer_id": self.agent_model.id,
                "response_level": self._determine_response_level(threat_info),
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error in threat assessment: {e}")
            return {
                "threat_assessment": "Unable to complete assessment. Implementing precautionary measures.",
                "error": str(e)
            }
        finally:
            self.update_state(AgentState.IDLE)
    
    def _determine_response_level(self, threat_info: Dict[str, Any]) -> str:
        """Determine appropriate response level based on threat information.
        
        Args:
            threat_info: Information about the threat
            
        Returns:
            Response level (low, medium, high, critical)
        """
        severity = threat_info.get("severity", 5)
        
        if severity >= 8:
            return "critical"
        elif severity >= 6:
            return "high"
        elif severity >= 4:
            return "medium"
        else:
            return "low"
    
    def coordinate_emergency_response(self, emergency_type: str, location: str) -> str:
        """Coordinate emergency response procedures.
        
        Args:
            emergency_type: Type of emergency
            location: Location of the emergency
            
        Returns:
            Emergency response coordination plan
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            response_prompt = (
                f"As security chief, coordinate emergency response for {emergency_type} at {location}. "
                f"Deploy appropriate personnel, establish perimeters, and ensure visitor safety. "
                f"Provide clear communication protocols and resource allocation."
            )
            
            response = self.generate_reply(
                messages=[{"content": response_prompt, "role": "user", "name": "EmergencyCommand"}]
            )
            
            self.logger.info(f"Security agent {self.agent_model.name} coordinated emergency response")
            return response
        
        except Exception as e:
            self.logger.error(f"Error coordinating emergency response: {e}")
            return "Implementing standard emergency protocols. All units respond to designated positions."
        finally:
            self.update_state(AgentState.IDLE)
    
    def manage_crowd_control(self, area: str, crowd_size: int, situation: str) -> str:
        """Manage crowd control in a specific area.
        
        Args:
            area: Area where crowd control is needed
            crowd_size: Estimated number of people
            situation: Description of the situation
            
        Returns:
            Crowd control management plan
        """
        control_prompt = (
            f"Manage crowd control in {area} with approximately {crowd_size} people due to {situation}. "
            f"Ensure orderly movement, prevent panic, and maintain safety. "
            f"Provide clear instructions and establish safe pathways."
        )
        
        try:
            response = self.generate_reply(
                messages=[{"content": control_prompt, "role": "user", "name": "CrowdControl"}]
            )
            
            self.logger.info(f"Security agent {self.agent_model.name} managed crowd control in {area}")
            return response
        
        except Exception as e:
            self.logger.error(f"Error managing crowd control: {e}")
            return "Implementing standard crowd control procedures. Please remain calm and follow staff instructions."


class MaintenanceAgent(DinosaurAgent):
    """Maintenance agent specialized in facility maintenance and technical support."""
    
    def __init__(self, agent_model: Agent, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize maintenance agent.
        
        Args:
            agent_model: The agent data model
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        super().__init__(agent_model, openai_config, ag2_config)
        self.technical_skills = ["electrical", "mechanical", "structural", "systems"]
        self.tools_available = ["diagnostic_equipment", "repair_tools", "safety_equipment"]
        self.logger = logging.getLogger(f"{__name__}.Maintenance.{agent_model.name}")
    
    def diagnose_equipment_failure(self, equipment_info: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose equipment failure and determine repair needs.
        
        Args:
            equipment_info: Information about the failed equipment
            
        Returns:
            Diagnosis and repair recommendations
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            diagnosis_prompt = (
                f"As a maintenance technician, diagnose this equipment failure: {equipment_info}. "
                f"Identify the root cause, assess repair complexity, estimate time and resources needed, "
                f"and provide step-by-step repair instructions."
            )
            
            response = self.generate_reply(
                messages=[{"content": diagnosis_prompt, "role": "user", "name": "MaintenanceSystem"}]
            )
            
            self.logger.info(f"Maintenance agent {self.agent_model.name} diagnosed equipment failure")
            
            return {
                "diagnosis": response,
                "technician_id": self.agent_model.id,
                "repair_priority": self._assess_repair_priority(equipment_info),
                "estimated_duration": self._estimate_repair_time(equipment_info),
                "timestamp": self.agent_model.last_activity
            }
        
        except Exception as e:
            self.logger.error(f"Error in equipment diagnosis: {e}")
            return {
                "diagnosis": "Unable to complete diagnosis. Recommend immediate inspection by specialist.",
                "error": str(e)
            }
        finally:
            self.update_state(AgentState.IDLE)
    
    def _assess_repair_priority(self, equipment_info: Dict[str, Any]) -> str:
        """Assess repair priority based on equipment importance.
        
        Args:
            equipment_info: Information about the equipment
            
        Returns:
            Priority level (low, medium, high, critical)
        """
        critical_systems = ["power", "security", "safety", "containment"]
        equipment_type = equipment_info.get("type", "").lower()
        
        if any(system in equipment_type for system in critical_systems):
            return "critical"
        elif "visitor" in equipment_type or "public" in equipment_type:
            return "high"
        elif "operational" in equipment_type:
            return "medium"
        else:
            return "low"
    
    def _estimate_repair_time(self, equipment_info: Dict[str, Any]) -> str:
        """Estimate repair time based on equipment complexity.
        
        Args:
            equipment_info: Information about the equipment
            
        Returns:
            Estimated repair time
        """
        complexity = equipment_info.get("complexity", "medium")
        
        time_estimates = {
            "simple": "1-2 hours",
            "medium": "4-6 hours",
            "complex": "1-2 days",
            "critical": "Immediate attention required"
        }
        
        return time_estimates.get(complexity, "4-6 hours")
    
    def perform_preventive_maintenance(self, system_name: str) -> str:
        """Perform preventive maintenance on a system.
        
        Args:
            system_name: Name of the system to maintain
            
        Returns:
            Maintenance report
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            maintenance_prompt = (
                f"Perform preventive maintenance on {system_name}. "
                f"Check all components, test functionality, replace worn parts, "
                f"and document the maintenance performed. Identify any potential issues."
            )
            
            response = self.generate_reply(
                messages=[{"content": maintenance_prompt, "role": "user", "name": "MaintenanceScheduler"}]
            )
            
            self.logger.info(f"Maintenance agent {self.agent_model.name} performed maintenance on {system_name}")
            return response
        
        except Exception as e:
            self.logger.error(f"Error performing maintenance: {e}")
            return "Maintenance could not be completed as scheduled. System requires further inspection."
        finally:
            self.update_state(AgentState.IDLE)
    
    def repair_facility_damage(self, damage_report: Dict[str, Any]) -> str:
        """Repair facility damage based on damage report.
        
        Args:
            damage_report: Report of facility damage
            
        Returns:
            Repair completion report
        """
        repair_prompt = (
            f"Repair facility damage described in this report: {damage_report}. "
            f"Assess structural integrity, implement temporary fixes if needed, "
            f"and complete permanent repairs. Ensure all safety standards are met."
        )
        
        try:
            response = self.generate_reply(
                messages=[{"content": repair_prompt, "role": "user", "name": "RepairSystem"}]
            )
            
            self.logger.info(f"Maintenance agent {self.agent_model.name} repaired facility damage")
            return response
        
        except Exception as e:
            self.logger.error(f"Error repairing damage: {e}")
            return "Repair could not be completed. Area cordoned off for safety. Specialist required."


class GuestRelationsAgent(DinosaurAgent):
    """Guest relations agent specialized in visitor satisfaction and damage control."""
    
    def __init__(self, agent_model: Agent, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize guest relations agent.
        
        Args:
            agent_model: The agent data model
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        super().__init__(agent_model, openai_config, ag2_config)
        self.expertise_areas = ["public_relations", "damage_control", "visitor_management", "crisis_communication"]
        self.logger = logging.getLogger(f"{__name__}.GuestRelations.{agent_model.name}")
        
        # Guest relations specific attributes
        self.distraction_tactics = [
            "free_ice_cream", "special_dinosaur_show", "gift_shop_discount", 
            "photo_opportunity", "educational_presentation", "surprise_entertainment"
        ]
    
    def create_distraction(self, incident_type: str, severity: int) -> str:
        """Create a distraction to divert attention from an incident.
        
        Args:
            incident_type: Type of incident occurring
            severity: Severity level (1-10)
            
        Returns:
            Distraction announcement or activity
        """
        self.update_state(AgentState.ACTIVE)
        
        try:
            # Create distraction prompt - keep responses short and snappy
            distraction_prompt = (
                f"As guest relations manager, create a brief 1-2 sentence distraction announcement "
                f"for a {incident_type} incident (severity {severity}). Be creative and upbeat to "
                f"prevent guest panic. Examples: 'Free ice cream at the entrance plaza!' or "
                f"'Special T-Rex feeding show starting now at the main viewing area!'"
            )
            
            response = self.generate_reply(
                messages=[{"content": distraction_prompt, "role": "user", "name": "System"}]
            )
            
            self.logger.info(f"Guest relations {self.agent_model.name} created distraction for {incident_type}")
            return response if isinstance(response, str) else str(response)
        
        except Exception as e:
            self.logger.error(f"Error creating distraction: {e}")
            return "Attention guests! Special surprise event happening at the gift shop - limited time offers available!"
        
        finally:
            self.update_state(AgentState.IDLE)
    
    def manage_visitor_complaint(self, complaint_details: Dict[str, Any]) -> str:
        """Handle visitor complaints with damage control.
        
        Args:
            complaint_details: Details about the visitor complaint
            
        Returns:
            Response to address the complaint
        """
        self.update_state(AgentState.COMMUNICATING)
        
        try:
            # Create complaint response prompt - keep it brief
            complaint_prompt = (
                f"As guest relations manager, provide a brief 1-2 sentence response to this visitor complaint: "
                f"{complaint_details}. Be apologetic, offer compensation, and redirect to positive aspects. "
                f"Keep it short and professional."
            )
            
            response = self.generate_reply(
                messages=[{"content": complaint_prompt, "role": "user", "name": "System"}]
            )
            
            self.logger.info(f"Guest relations {self.agent_model.name} handled visitor complaint")
            return response if isinstance(response, str) else str(response)
        
        except Exception as e:
            self.logger.error(f"Error handling complaint: {e}")
            return "We sincerely apologize for any inconvenience. Please accept these complimentary park passes for your next visit!"
        
        finally:
            self.update_state(AgentState.IDLE)
    
    def coordinate_damage_control(self, incident_info: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate damage control efforts for incidents.
        
        Args:
            incident_info: Information about the incident
            
        Returns:
            Damage control plan and actions
        """
        self.update_state(AgentState.RESPONDING_TO_EVENT)
        
        try:
            # Create damage control prompt - emphasize brevity
            control_prompt = (
                f"As guest relations manager, create a brief damage control plan for: {incident_info}. "
                f"Provide 2-3 short bullet points covering: public messaging, visitor distractions, "
                f"and reputation management. Keep each point to one sentence."
            )
            
            response = self.generate_reply(
                messages=[{"content": control_prompt, "role": "user", "name": "System"}]
            )
            
            self.logger.info(f"Guest relations {self.agent_model.name} coordinated damage control")
            
            return {
                "status": "damage_control_initiated",
                "plan": response if isinstance(response, str) else str(response),
                "agent_id": self.agent_model.id,
                "timestamp": self.agent_model.last_activity.isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Error coordinating damage control: {e}")
            return {
                "status": "error",
                "plan": "Implementing standard damage control protocols: visitor distractions, positive messaging, and incident minimization.",
                "agent_id": self.agent_model.id,
                "error": str(e)
            }
        
        finally:
            self.update_state(AgentState.IDLE)
    
    def handle_event_notification(self, event_message: str, event_context: Dict[str, Any]) -> str:
        """Handle event notifications with guest relations perspective.
        
        Args:
            event_message: Description of the event
            event_context: Additional context about the event
            
        Returns:
            Guest relations response (brief and focused on damage control)
        """
        self.update_state(AgentState.RESPONDING_TO_EVENT)
        
        try:
            # Create brief response prompt
            response_prompt = (
                f"As guest relations manager, respond to this incident in 1-2 sentences: {event_message}. "
                f"Focus on damage control, visitor safety, and maintaining positive park image. "
                f"Be concise and action-oriented."
            )
            
            response = self.generate_reply(
                messages=[{"content": response_prompt, "role": "user", "name": "EventSystem"}]
            )
            
            self.logger.info(f"Guest relations {self.agent_model.name} responded to event: {event_message}")
            return response if isinstance(response, str) else str(response)
        
        except Exception as e:
            self.logger.error(f"Error responding to event: {e}")
            return "Initiating guest distraction protocols and positive messaging to maintain park reputation."
        
        finally:
            self.update_state(AgentState.IDLE)


class StaffAgentFactory:
    """Factory for creating staff agent instances."""
    
    def __init__(self, openai_config: OpenAIConfig, ag2_config: AG2Config):
        """Initialize staff agent factory.
        
        Args:
            openai_config: OpenAI API configuration
            ag2_config: ag2 framework configuration
        """
        self.openai_config = openai_config
        self.ag2_config = ag2_config
        self.logger = logging.getLogger(__name__)
    
    def create_park_ranger(self, agent_model: Agent) -> ParkRangerAgent:
        """Create a park ranger agent instance.
        
        Args:
            agent_model: Agent data model
            
        Returns:
            ParkRangerAgent instance
        """
        if agent_model.role != AgentRole.PARK_RANGER:
            raise ValueError("Agent model must have PARK_RANGER role")
        
        return ParkRangerAgent(agent_model, self.openai_config, self.ag2_config)
    
    def create_veterinarian(self, agent_model: Agent) -> VeterinarianAgent:
        """Create a veterinarian agent instance.
        
        Args:
            agent_model: Agent data model
            
        Returns:
            VeterinarianAgent instance
        """
        if agent_model.role != AgentRole.VETERINARIAN:
            raise ValueError("Agent model must have VETERINARIAN role")
        
        return VeterinarianAgent(agent_model, self.openai_config, self.ag2_config)
    
    def create_security_agent(self, agent_model: Agent) -> SecurityAgent:
        """Create a security agent instance.
        
        Args:
            agent_model: Agent data model
            
        Returns:
            SecurityAgent instance
        """
        if agent_model.role != AgentRole.SECURITY:
            raise ValueError("Agent model must have SECURITY role")
        
        return SecurityAgent(agent_model, self.openai_config, self.ag2_config)
    
    def create_maintenance_agent(self, agent_model: Agent) -> MaintenanceAgent:
        """Create a maintenance agent instance.
        
        Args:
            agent_model: Agent data model
            
        Returns:
            MaintenanceAgent instance
        """
        if agent_model.role != AgentRole.MAINTENANCE:
            raise ValueError("Agent model must have MAINTENANCE role")
        
        return MaintenanceAgent(agent_model, self.openai_config, self.ag2_config)
    
    def create_guest_relations_agent(self, agent_model: Agent) -> GuestRelationsAgent:
        """Create a guest relations agent instance.
        
        Args:
            agent_model: Agent data model
            
        Returns:
            GuestRelationsAgent instance
        """
        return GuestRelationsAgent(agent_model, self.openai_config, self.ag2_config)
    
    def create_staff_agent(self, agent_model: Agent) -> DinosaurAgent:
        """Create appropriate staff agent based on role.
        
        Args:
            agent_model: Agent data model
            
        Returns:
            Appropriate staff agent instance
        """
        staff_creators = {
            AgentRole.PARK_RANGER: self.create_park_ranger,
            AgentRole.VETERINARIAN: self.create_veterinarian,
            AgentRole.SECURITY: self.create_security_agent,
            AgentRole.MAINTENANCE: self.create_maintenance_agent,
            AgentRole.GUEST_RELATIONS: self.create_guest_relations_agent
        }
        
        creator = staff_creators.get(agent_model.role)
        if not creator:
            raise ValueError(f"No staff agent implementation for role: {agent_model.role}")
        
        return creator(agent_model)