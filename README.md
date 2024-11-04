
### **WARNING : This projet is a work in progress, onging changes are likely to break some of the functionalities**

  

# Cyber Multi-Agent System Development Environment

**Cyber Multi-Agent System Development Environment (CybMASDE)** is intended to implement a Dec-POMDP model of any environment where agents have to collaborate to reach a global goal under extra constraints. CybMASDE is mostly intended to model a Cyberdefense scenario dealing with a network on which attack and defender agents are interacting each other.

  

## Project Goals

This project is largely inspired by the context of [CyberBattleSim](https://github.com/microsoft/CyberBattleSim) in a multi-agent approach.

It aims to provide a way to simulate a network of nodes on which attack actions are coordinately applied by attacking agents according to a realistic based attack scenario.

Additionally, like cyber-attackers, it also aims to implement defenders whose behaviors result in different organization (whether pre-determined or not).

Consequences of applied action brings out a new environment state, optionally changing the agent knowledge and observations and bringing agents closer or further from their local goals.

  

## Requirements

- Python version >= 3.9

- Node version >= v18.14.x

  

## Installation

In "backend", type: "python install requirements.txt"

Then, in "backend/src", type: "python -m flask --app server run"

  

In "frontend", type "npm install"

Then, in "frontend", type "npm run start"

  

Open a webpage at http://localhost:4200/

The user interface should be displayed

  

## Basic interface usage

  

When using terminal, avaialbe command lines are

  

- ls : list avaialbe environment scenarios

- load [environment_file.json]

  

![alt text](https://github.com/julien6/CybMASDE/blob/main/blop/basics.gif?raw=true)

  

- next : so next agent plays to interact with environment / other agents

  

![alt text](https://github.com/julien6/CybMASDE/blob/main/blop/next.gif?raw=true)

  

- iterate_over [number of iteration] : so several iteration can occur in a single command

  

![alt text](https://github.com/julien6/CybMASDE/blob/main/blop/iterate_over.gif?raw=true)

## Environment creation

An simulation saving file describes:
 - The nodes (such as firewalls, workstation, server...) with their properties including deployed agents' ones as well :

*Nodes environement skeleton :*
```json
{
	"meta_data": {},
	"nodes_properties": {
		"employee_workstation": {
			"installed_operating_system": "Windows/12",
			"installed_softwares": "MSOffice/2021",
			...
			"processes": {
				"agents": {
					...
				}
			}
			...
		},
		"db_server": {
			...
		}
	},
	"actions": {
		...
	}
}
```
*Example of agents :*
```json
{
	"attacker1": {
		"behaviour": "idle",
		"observations": {
			"found_password_file": "pwd.txt",
			...
		},
		"running": true,
		"binary_file_location": "C:\\Users\\mwlr.exe",
		...
	},
	"defender1": {
		"behaviour": "idle",
		"observations": {
			"is_anomaly_in_log": true,
			...
		},
		"running": true,
		"root_privilege": "root",
		"binary_file_location": "C:\\Users\\dfdr.exe"
		...
	}
}
```

 - The actions to interact with nodes

```json
"example_Action": {
    "cost": 15,
    "description": "Example action description",
    "precondition": "({{agent}}.property_id1.property_id1_1 == 'v1' and {{node}}.id2 == 'v2) or {{include(precondition_file)}}",
    "postcondition": {
        "{{agent}}.knowledge.reimagable": "{{node}}.reimagable",
        "{{node}}.logs.{{last_index}}": "'{{agent}} observed \"reimagable\" of {{node}} at {{current_time}}'"
    },
    "success_probability": 1
}
```


A full example is given below :
```json
{
	"meta_data": {},
	"nodes_properties": {
		"node1_id": {
			"installed_operating_system": "Windows/12",
			"installed_softwares": "MSOffice/2021",
			...
			"processes": {
				"agents": {
					"attacker1": {
						"behaviour": "idle",
						"observations": {
							"found_password_file": "pwd.txt",
							...
						},
						"running": true,
						"binary_file_location": "C:\\Users\\mwlr.exe",
						...
					},
					"defender1": {
						"behaviour": "idle",
						"observations": {
							"is_anomaly_in_log": true,
							...
						},
						"running": true,
						"root_privilege": "root",
						"binary_file_location": "C:\\Users\\dfdr.exe"
						...
					}
				}
				...
			}
		},
		"node2_id": {
			...
		}
	},
	"actions": {
		"example_action": {
	        "cost": 15,
	        "description": "Example action description",
	        "precondition": "({{agent}}.property_id1.property_id1_1 == 'v1' and {{node}}.id2 == 'v2) or {{include(precondition_file)}}",
	        "postcondition": {
	            "{{agent}}.knowledge.reimagable": "{{node}}.reimagable",
	            "{{node}}.logs.{{last_index}}": "'{{agent}} observed \"reimagable\" of {{node}} at {{current_time}}'"
	        },
	        "success_probability": 1
        }
	}
}
```
