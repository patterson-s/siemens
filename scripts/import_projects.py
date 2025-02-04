import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.models import Project

def import_projects_from_csv():
    print("Starting project import process...")
    
    # Check if CSV file exists
    csv_path = 'data/SI Country Context v2 sample.csv'
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {os.path.abspath(csv_path)}")
        return
    
    print(f"Reading CSV file from {csv_path}")
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Print column names to verify
    print("\nCSV columns:")
    print(df.columns.tolist())
    
    # Print the first few rows to verify data
    print("\nFirst few rows of CSV:")
    print(df.head())
    print(f"\nTotal rows in CSV: {len(df)}")
    
    # Clean up the data
    df = df.fillna('')  # Replace NaN with empty string
    
    # Convert funding amount to numeric, removing any currency symbols and commas
    df['Funding Amount (millions of USD)'] = pd.to_numeric(
        df['Funding Amount (millions of USD)'].replace('[\$,]', '', regex=True), 
        errors='coerce'
    )
    
    # Create Flask app context
    print("\nCreating Flask app context...")
    app = create_app()
    with app.app_context():
        # Delete existing projects
        print("Clearing existing projects...")
        existing_count = db.session.query(Project).count()
        print(f"Found {existing_count} existing projects")
        db.session.query(Project).delete()
        
        # Import each row
        print("\nImporting new projects...")
        for index, row in df.iterrows():
            print(f"\nProcessing row {index + 1}:")
            print(f"  Name of Round: {row['Name of Round']}")
            print(f"  Partner Name: {row['Integrity Partner Name']}")
            
            project = Project(
                name_of_round=float(row['Name of Round']),
                name=row['Integrity Partner Name'],  # Using Partner Name as project name
                file_number_db=row['File Number (in DB)'],
                scope=row['Scope'],
                region=row['Region '],  # Note the space after Region
                countries_covered=row['Countries Covered'],
                integrity_partner_name=row['Integrity Partner Name'],
                partner_type=row['Partner Type'],
                project_partners=row['Project Partners (if applicable)'],
                wb_or_eib=row['WB or EIB?'],
                key_project_objectives=row['Key Project Objectives'],
                sectoral_scope=row['Sectoral Scope'],
                specific_sector=row['Specific Sector'],
                funding_amount_usd=row['Funding Amount (millions of USD)'],
                duration=row['Duration (Intended)'],
                start_year=int(row['Start and End Years (Actual)'].split('-')[0]) if row['Start and End Years (Actual)'] else None,
                end_year=int(row['Start and End Years (Actual)'].split('-')[1]) if row['Start and End Years (Actual)'] else None,
                wb_income_classification=row['WB Income Classification'],
                cci=row['CCI'] if row['CCI'] != '' else None,
                government_type_eiu=row['Government Type (EIU)'],
                government_score_eiu=row['Government Score (EIU)'] if row['Government Score (EIU)'] != '' else None,
                active=True
            )
            db.session.add(project)
            print(f"  Added project to session")
        
        # Commit the changes
        try:
            print("\nCommitting changes to database...")
            db.session.commit()
            print("Successfully imported projects")
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred during commit: {str(e)}")
            raise e

if __name__ == '__main__':
    print("Starting import script...")
    import_projects_from_csv() 