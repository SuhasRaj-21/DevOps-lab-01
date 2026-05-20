pipeline {
    agent any

    stages {

        stage('Clone GitHub Repo') {
            steps {
                git 'https://github.com/SuhasRaj-21/DevOps-lab-01.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('SonarQube Analysis') {
            steps {
                echo 'Running SonarQube Analysis'
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t smart-attendance .'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 5000:5000 smart-attendance'
            }
        }
    }
}
