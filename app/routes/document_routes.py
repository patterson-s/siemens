@bp.route('/upload', methods=['POST'])
@login_required
def upload_document():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Process the file
        filename = secure_filename(file.filename)
        project_id = request.form.get('project_id')
        document_type = request.form.get('document_type')
        
        # Extract content from the file
        content = extract_text_from_file(file)
        
        # Create a preview
        content_preview = content[:500] + "..." if len(content) > 500 else content
        
        # Create the document
        document = Document(
            project_id=project_id,
            filename=filename,
            file_type=file.filename.rsplit('.', 1)[1].lower(),
            document_type=document_type,
            file_size=len(content.split()),  # Word count
            content_preview=content_preview
        )
        
        # Set the encrypted content
        document.set_content(content)
        
        db.session.add(document)
        db.session.commit()
        
        flash('Document uploaded successfully!', 'success')
        return redirect(url_for('documents.view', id=document.id))
    
    flash('File type not allowed', 'danger')
    return redirect(request.url) 